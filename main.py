
import cv2
import time
import os
import sys
import argparse
import io
import numpy as np


try:
    from urllib.request import urlopen, Request as UrlRequest
except ImportError:
    from urllib2 import urlopen, Request as UrlRequest

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Faltan dependencias. Ejecuta:")
    print("        pip install opencv-python Pillow numpy")
    sys.exit(1)


UMBRAL_MOVIMIENTO   = 1500
BLUR_KERNEL         = (21, 21)
UMBRAL_BINARIO      = 5
COOLDOWN_SEGUNDOS   = 0.5
VELOCIDAD_GIF_MS    = 80

FRAME_ESTATICO      = 0
FRAME_INICIO_LOOP   = 5
FRAME_FIN_LOOP      = 15


class GestorGIF(object):
    def __init__(self, fuente_gif, dimension):
        self.fuente_gif = fuente_gif
        self.dimension = dimension
        self._frames = []
        self._idx = FRAME_ESTATICO
        self._ultimo_cambio = 0.0
        self._reproduciendo_entero = True 
        self._precargar_frames()

    def _precargar_frames(self):
        try:
            fuente = self.fuente_gif
            if fuente.lower().startswith("http://") or fuente.lower().startswith("https://"):
                print("[GIF] Descargando desde URL...")
                req = UrlRequest(fuente, headers={"User-Agent": "Mozilla/5.0"})
                datos = urlopen(req, timeout=15).read()
                gif = Image.open(io.BytesIO(datos))
                print("[GIF] Descarga completa ({} bytes).".format(len(datos)))
            else:
                gif = Image.open(fuente)

            while True:
                frame_rgb = gif.copy().convert("RGB")
                frame_np = np.array(frame_rgb)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frame_bgr = cv2.resize(frame_bgr, self.dimension)
                
                self._frames.append(frame_bgr)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        except Exception as e:
            print("[ERROR] No se pudo cargar el GIF:", e)
            self._frames =[]

        if self._frames:
            print("[GIF] {} frames cargados y redimensionados correctamente a {}x{}.".format(
                len(self._frames), self.dimension[0], self.dimension[1]))

    def obtener_frame(self, animar):
        """Calcula cuál es el frame actual y lo devuelve para ser ensamblado."""
        if not self._frames:
            return np.zeros((self.dimension[1], self.dimension[0], 3), dtype=np.uint8)

        tope = len(self._frames) - 1
        estatico = min(FRAME_ESTATICO, tope)
        inicio = min(FRAME_INICIO_LOOP, tope)
        fin = min(FRAME_FIN_LOOP, tope)
        
        if inicio > fin:
            inicio, fin = fin, inicio

        if animar:
            ahora = time.time()
            if (ahora - self._ultimo_cambio) * 1000 > VELOCIDAD_GIF_MS:
                
                if self._reproduciendo_entero:
                    if self._idx < tope:
                        self._idx += 1
                    else:
                        self._reproduciendo_entero = False
                        self._idx = inicio
                else:
                    if self._idx < inicio or self._idx >= fin:
                        self._idx = inicio
                    else:
                        self._idx += 1
                
                self._ultimo_cambio = ahora
        else:
            self._idx = estatico
            self._reproduciendo_entero = True
            self._ultimo_cambio = time.time()

        return self._frames[self._idx]



class DetectorMovimiento(object):

    def __init__(self, indice_camara, fuente_gif):
        self.cap = cv2.VideoCapture(indice_camara)
        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara {}.".format(indice_camara))
            sys.exit(1)

        ret, frame_prueba = self.cap.read()
        if ret:
            alto, ancho = frame_prueba.shape[:2]
        else:
            alto, ancho = 480, 640 
            
        dimension_camara = (ancho, alto)

        self.gestor_gif = GestorGIF(fuente_gif, dimension_camara)
        self.frame_anterior = None
        self.ultimo_movimiento = 0.0
        self.debug = False

        self.nombre_ventana = "perrito feliz cuando llegas :3"
        cv2.namedWindow(self.nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.nombre_ventana, ancho * 2, alto)

    def _preprocesar(self, frame):
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, BLUR_KERNEL, 0)
        return gris

    def _calcular_diferencia(self, actual):
        delta = cv2.absdiff(self.frame_anterior, actual)
        _, mascara = cv2.threshold(delta, UMBRAL_BINARIO, 255, cv2.THRESH_BINARY)
        mascara = cv2.dilate(mascara, None, iterations=2)
        return mascara

    def ejecutar(self):
        print("═" * 55)
        print("  Detector de movimiento activo")
        print("  Teclas: [Q/ESC] salir  |  [D] debug mask")
        print("═" * 55)

        while True:
            ret, frame_camara = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            actual = self._preprocesar(frame_camara)

            if self.frame_anterior is None:
                self.frame_anterior = actual
                continue

            mascara = self._calcular_diferencia(actual)
            pixeles_blancos = cv2.countNonZero(mascara)
            hay_movimiento = pixeles_blancos > UMBRAL_MOVIMIENTO

            if hay_movimiento:
                self.ultimo_movimiento = time.time()
                
                contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contornos:
                    if cv2.contourArea(c) < 500:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame_camara, (x, y), (x + w, y + h), (0, 0, 255), 2)

            debe_animarse = (time.time() - self.ultimo_movimiento) <= COOLDOWN_SEGUNDOS
            
            frame_gif = self.gestor_gif.obtener_frame(animar=debe_animarse)
            estado = "MOVIMIENTO" if hay_movimiento else "Pausado"
            color  = (0, 0, 255) if hay_movimiento else (0, 200, 0)
            cv2.putText(frame_camara, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame_camara, "[Q] salir  [D] debug", (10, frame_camara.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            pantalla_dividida = np.hstack((frame_camara, frame_gif))

            cv2.imshow(self.nombre_ventana, pantalla_dividida)

            if self.debug:
                cv2.imshow("Mascara de Deteccion", mascara)
            else:
                try:
                    if cv2.getWindowProperty("Mascara de Deteccion", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow("Mascara de Deteccion")
                except Exception:
                    pass

            self.frame_anterior = actual

            tecla = cv2.waitKey(30) & 0xFF
            if tecla == ord('q') or tecla == 27:
                break
            elif tecla == ord('d'):
                self.debug = not self.debug

        self._limpiar()

    def _limpiar(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detector cerrado.")

def main():
    parser = argparse.ArgumentParser(description="Detecta movimiento con la webcam y reproduce un GIF con lógica avanzada.")
    parser.add_argument("--gif", required=True, help="URL o ruta local del GIF")
    parser.add_argument("--cam", type=int, default=0, help="Índice de cámara (default: 0)")
    args = parser.parse_args()

    es_url = args.gif.lower().startswith("http://") or args.gif.lower().startswith("https://")

    if not es_url and not os.path.isfile(args.gif):
        print("[ERROR] No se encontró el archivo local: '{}'".format(args.gif))
        sys.exit(1)

    detector = DetectorMovimiento(indice_camara=args.cam, fuente_gif=args.gif)
    detector.ejecutar()

if __name__ == "__main__":
    main()