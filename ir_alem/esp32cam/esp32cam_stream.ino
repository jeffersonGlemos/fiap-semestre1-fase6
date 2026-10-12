/*
 * esp32cam_stream.ino: ESP32-CAM (AI-Thinker) transmitindo MJPEG por Wi-Fi.
 *
 * Parti do exemplo CameraWebServer e mantive apenas o que o cliente_vscode.py consome.
 * A ESP32-CAM não tem memória para rodar o YOLOv5 do projeto, então nesta placa ela atua só
 * como câmera Wi-Fi e a inferência acontece no PC (modo BASE).
 * Rotas (porta 80), iguais às do camera_server.py:
 *   /stream     vídeo MJPEG contínuo
 *   /frame.jpg  um único quadro
 *   /saude      status em JSON
 *
 * Arduino IDE: placa "AI Thinker ESP32-CAM", Partition Scheme "Huge APP", PSRAM habilitada.
 * Após a gravação, o Monitor Serial (115200) mostra o IP; no PC, o cliente é iniciado com:
 *   python cliente_vscode.py --fonte http://<IP>/stream --modelo best.pt
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <time.h>
#include <sys/time.h>

// ---- Credenciais da rede: valores de exemplo, substituídos antes da gravação (senhas reais ficam fora do repositório) ----
const char *WIFI_SSID  = "SUA_REDE";
const char *WIFI_SENHA = "SUA_SENHA";

// ---- Pinagem da câmera OV2640 na placa AI-Thinker ----
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#define FRONTEIRA "quadro"
static const char *TIPO_STREAM = "multipart/x-mixed-replace;boundary=" FRONTEIRA;
static const char *SEPARADOR = "\r\n--" FRONTEIRA "\r\n";
// X-Timestamp em epoch (sincronizado por NTP): o cliente usa o valor para medir a latência da captura até a tela
static const char *CABECALHO = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %ld.%06ld\r\n\r\n";

httpd_handle_t servidor = NULL;

bool iniciarCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM; c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;       // o sensor já entrega JPEG, então o ESP32 apenas repassa os bytes
  c.frame_size = FRAMESIZE_VGA;          // 640x480: próximo da entrada de 640 px do YOLO, sem desperdiçar banda
  c.jpeg_quality = 12;                   // escala de 0 a 63; quanto menor, melhor a qualidade
  c.fb_count = 2;                        // dois buffers na PSRAM permitem capturar e enviar em paralelo
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;      // descarta quadros antigos para reduzir a latência
  return esp_camera_init(&c) == ESP_OK;
}

// /stream: envia quadros continuamente até o cliente desconectar
static esp_err_t rotaStream(httpd_req_t *req) {
  char cab[128];
  httpd_resp_set_type(req, TIPO_STREAM);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return ESP_FAIL;
    struct timeval agora;
    gettimeofday(&agora, NULL);
    size_t n = snprintf(cab, sizeof(cab), CABECALHO, fb->len, agora.tv_sec, agora.tv_usec);
    esp_err_t r = httpd_resp_send_chunk(req, SEPARADOR, strlen(SEPARADOR));
    if (r == ESP_OK) r = httpd_resp_send_chunk(req, cab, n);
    if (r == ESP_OK) r = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);            // devolve o buffer para a câmera reutilizá-lo
    if (r != ESP_OK) return r;           // o cliente fechou a conexão
  }
}

// /frame.jpg: um único quadro
static esp_err_t rotaFrame(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return httpd_resp_send_500(req);
  httpd_resp_set_type(req, "image/jpeg");
  esp_err_t r = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return r;
}

// /saude: status usado pelo cliente para medir o RTT e acompanhar o sinal Wi-Fi (RSSI)
static esp_err_t rotaSaude(httpd_req_t *req) {
  char json[160];
  snprintf(json, sizeof(json),
           "{\"servidor\":\"esp32cam\",\"inferencia\":\"nenhuma\",\"rssi_dbm\":%d,\"timestamp\":%ld}",
           WiFi.RSSI(), (long)time(NULL));
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

void iniciarServidor() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 80;
  httpd_uri_t rotas[] = {
    {"/stream", HTTP_GET, rotaStream, NULL},
    {"/frame.jpg", HTTP_GET, rotaFrame, NULL},
    {"/saude", HTTP_GET, rotaSaude, NULL},
  };
  if (httpd_start(&servidor, &cfg) == ESP_OK)
    for (auto &r : rotas) httpd_register_uri_handler(servidor, &r);
}

void setup() {
  Serial.begin(115200);
  if (!iniciarCamera()) {
    Serial.println("Falha ao iniciar a camera (confira PSRAM e o modelo da placa).");
    return;
  }
  WiFi.begin(WIFI_SSID, WIFI_SENHA);
  WiFi.setSleep(false);                  // desativa a economia de energia do rádio para estabilizar o stream
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  configTime(-3 * 3600, 0, "pool.ntp.org");  // sincroniza o relógio em epoch para o X-Timestamp
  iniciarServidor();
  Serial.printf("\nStream pronto: http://%s/stream  (RSSI %d dBm)\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
}

void loop() {
  delay(10000);                          // todo o trabalho acontece nos handlers das rotas HTTP
}
