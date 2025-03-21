# Kütüphaneler
import cv2
from cvzone.HandTrackingModule import HandDetector
from time import time
import numpy as np
from pynput.keyboard import Controller

# Buton sınıfı
class Button():
    def __init__(self, pos, text, size=[80, 80]):
        self.pos = pos
        self.text = text
        self.size = size

# Butonları çizen fonksiyon
def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

# Kamera ayarları
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Hata: Kamera açılamadı.")
    exit()

cap.set(3, 1280)
cap.set(4, 720)

# El algılama nesnesi
detector = HandDetector(detectionCon=0.8, maxHands=2)
keyboard = Controller()

# Sanal klavye tuş düzeni
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
buttonList = [Button([100 + 100 * j, 100 + 100 * i], key) for i in range(len(keys)) for j, key in enumerate(keys[i])]
finalText = ""

# Her buton için son tıklama zamanını tutacak bir sözlük
lastClickTime = {button.text: 0 for button in buttonList}

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Hata: Kameradan görüntü alınamadı.")
        break

    img = cv2.flip(img, 1)  # Ayna etkisi için görüntüyü yatay çevir

    # Elleri algıla
    hands, img = detector.findHands(img, draw=True)

    # Butonları çiz
    img = drawAll(img, buttonList)

    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            if len(lmList) >= 13:
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size

                    # İşaret parmağı ucu butonun içinde mi?
                    if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

                        # Tüm parmaklar aşağıda mı ve sadece işaret parmağı yukarıda mı?
                        fingers = detector.fingersUp(hand)
                        if fingers == [0, 1, 0, 0, 0]:  # Sadece işaret parmağı yukarıda
                            currentTime = time()
                            if currentTime - lastClickTime[button.text] > 0.5:  # 0.5 saniye bekleme
                                keyboard.press(button.text)
                                cv2.rectangle(img, button.pos, (x + w, y + h), (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                                finalText += button.text
                                lastClickTime[button.text] = currentTime  # Son tıklama zamanını güncelle

    # Label renkleri ve label 
    cv2.rectangle(img, (50, 20), (1230, 80), (0, 0, 0), cv2.FILLED)  # Siyah arka plan
    cv2.putText(img, finalText, (60, 65), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)  # Beyaz yazı

    # Görüntüyü göster
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()