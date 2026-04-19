#include <WiFi.h>

const char* SSID     = "Ashitabh";
const char* PASSWORD = "ashitabh";

WiFiServer server(8080);

#define TOTAL_BYTES (120 * 10)  // 1200

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

  uint8_t buf[TOTAL_BYTES];
  int total_received = 0;

  // Keep reading until we have all 1200 bytes or client disconnects
  while (client.connected() && total_received < TOTAL_BYTES) {
    if (client.available()) {
      int chunk = client.read(buf + total_received, TOTAL_BYTES - total_received);
      if (chunk > 0) total_received += chunk;
    }
  }

  Serial.print("Total bytes received: ");
  Serial.println(total_received);

  // Print first row (10 values) as a sanity check
  Serial.print("First 10 values: ");
  for (int i = 0; i < 10 && i < total_received; i++) {
    Serial.print(buf[i]);
    Serial.print(" ");
  }
  Serial.println();

  // Send back the count as 2 bytes (big-endian) so Python can confirm
  uint8_t reply[2];
  reply[0] = (total_received >> 8) & 0xFF;
  reply[1] =  total_received       & 0xFF;
  client.write(reply, 2);

  client.stop();
  Serial.println("Client disconnected");
}