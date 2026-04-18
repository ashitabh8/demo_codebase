#include <WiFi.h>

const char* SSID = "Ashitabh";
const char* PASSWORD = "ashitabh";

WiFiServer server(8080);

void setup() {
  Serial.begin(115200);
  delay(2000);

  WiFi.begin(SSID, PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("Server ready");
}

void loop() {
  WiFiClient client = server.available();
  if (!client) return;

  Serial.println("Client connected");

  uint8_t buf[3];
  int count = 0;

  while (client.connected() && count < 3) {
    if (client.available()) {
      buf[count] = client.read();
      count++;
    }
  }

  if (count == 3) {
    Serial.print("Received: ");
    Serial.print(buf[0]);
    Serial.print(" ");
    Serial.print(buf[1]);
    Serial.print(" ");
    Serial.println(buf[2]);

    for (int i = 0; i < 3; i++) {
      buf[i] = buf[i] + 10;
    }

    client.write(buf, 3);

    Serial.print("Sent back: ");
    Serial.print(buf[0]);
    Serial.print(" ");
    Serial.print(buf[1]);
    Serial.print(" ");
    Serial.println(buf[2]);
  }

  client.stop();
  Serial.println("Client disconnected");
}