import numpy as np
from dataclasses import dataclass
import pyrealsense2 as rs
class CameraIntrinsics:
    """
    Parametri intrinseci della camera (modello pinhole).
    
    Attributi:
        fx: lunghezza focale lungo l'asse x (in pixel)
        fy: lunghezza focale lungo l'asse y (in pixel)
        cx: coordinata x del punto principale (centro ottico) in pixel
        cy: coordinata y del punto principale (centro ottico) in pixel
    """
    
    def __init__(self, device_id: str) -> None:
        """
        Inizializza i parametri intrinseci recuperandoli dal sensore RealSense.
        
        Parametri:
            device_id: ID/serial number del dispositivo RealSense
        """
        # Recupera gli intrinsics dal sensore, non ho necessita di intrinsic quindi ignoro il primo elemento della tupla restituita
        _, self.fx, self.fy, self.cx, self.cy = self._fetch_intrinsics_from_device(device_id)
        
        # Validazione dei parametri intrinseci
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"Le lunghezze focali devono essere positive: fx={self.fx}, fy={self.fy}"
            )
        if self.cx < 0 or self.cy < 0:
            raise ValueError(
                f"Il punto principale deve avere coordinate non negative: cx={self.cx}, cy={self.cy}"
            )

    @staticmethod
    def _fetch_intrinsics_from_device(device_id: str) -> tuple:
        """
        Recupera i parametri intrinseci dal sensore RealSense.
        
        Parametri:
            device_id: ID/serial number del dispositivo RealSense
            
        Restituisce:
            Tupla (intrinsics_obj, fx, fy, cx, cy)
        """
        if not device_id:
            # raise ValueError("Il device_id non può essere vuoto. Fornire un ID valido del dispositivo RealSense.")
            return None, 608.1531982421875, 608.2805786132812, 315.12713623046875, 259.9116516113281
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            profile = pipeline.start(config)
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

            pipeline.stop()
        except:
            raise RuntimeError(
                f"Impossibile recuperare gli intrinseci dal dispositivo con ID '{device_id}'. "
                "Uso valori default per test."
            )


        return intrinsics, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

    def to_matrix(self) -> np.ndarray:
        """
        Restituisce la matrice intrinseca K (3x3).
        
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0    ]
        ], dtype=np.float64)


def build_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:

    # Precalcolo dei seni e coseni per efficienza e leggibilità
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Matrice di rotazione attorno all'asse X (roll)
    Rx = np.array([
        [1.0, 0.0,  0.0],
        [0.0, cr,  -sr ],
        [0.0, sr,   cr ]
    ], dtype=np.float64)

    # Matrice di rotazione attorno all'asse Y (pitch)
    Ry = np.array([
        [ cp, 0.0, sp ],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp ]
    ], dtype=np.float64)

    # Matrice di rotazione attorno all'asse Z (yaw)
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    R = Rz @ Ry @ Rx

    # Verifica che la matrice risultante sia una rotazione valida
    # (determinante ≈ 1 e R^T @ R ≈ I)
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(
            f"La matrice risultante non è una rotazione valida (det={det:.6f})"
        )

    return R


def build_T_cam_to_waist(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    
    # Validazione delle dimensioni di R
    if R.shape != (3, 3):
        raise ValueError(
            f"R deve essere 3x3, ricevuta forma {R.shape}"
        )

    # Conversione del vettore di traslazione in forma (3,)
    t = np.asarray(t, dtype=np.float64).flatten()
    if t.shape != (3,):
        raise ValueError(
            f"t deve avere 3 elementi, ricevuti {t.shape[0]}"
        )

    # Costruzione della matrice 4x4
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R        # Blocco rotazione (in alto a sinistra)
    T[:3, 3] = t          # Blocco traslazione (colonna destra)
    # L'ultima riga [0, 0, 0, 1] è già impostata da np.eye

    return T


def deproject_pixel(
    u: float,
    v: float,
    depth_mm: float,
    intrinsics: CameraIntrinsics
) -> np.ndarray:
    """
    Converte un pixel (u, v) con profondità in millimetri in un punto 3D
    nel sistema di riferimento della camera.
    
    Modello pinhole inverso:
        z = depth_mm / 1000.0          (conversione mm -> metri)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    
    Il sistema di riferimento della camera segue la convenzione standard:
        - Z: avanti (direzione di vista)
        - X: destra
        - Y: basso
    
    Parametri:
        u: coordinata orizzontale del pixel (colonna)
        v: coordinata verticale del pixel (riga)
        depth_mm: profondità in millimetri (dalla mappa di profondità)
        intrinsics: parametri intrinseci della camera
    
    Restituisce:
        point_cam: punto 3D nel frame camera [x, y, z] in metri
    """
    # Controllo che la profondità sia valida
    if depth_mm <= 0:
        raise ValueError(
            f"La profondità deve essere positiva, ricevuto: {depth_mm} mm. "
            "Un valore nullo o negativo indica un pixel senza misura valida."
        )

    # Controllo di sicurezza per profondità troppo elevate (possibile errore sensore)
    MAX_DEPTH_MM = 20000.0  # 20 metri: limite ragionevole per RealSense
    if depth_mm > MAX_DEPTH_MM:
        raise ValueError(
            f"Profondità sospettamente alta: {depth_mm} mm (max consentito: {MAX_DEPTH_MM} mm). "            
            "Verificare l'unità di misura o la qualità del dato."
        )
    # Conversione da millimetri a metri    
    z = depth_mm / 1000.0
    # Deproiezione usando il modello pinhole inverso    
    x = (u - intrinsics.cx) * z / intrinsics.fx    
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=np.float64)

def camera_point_trasformation(P_c: np.ndarray) -> np.ndarray:
    # Invertiamo i punti per poter fare rotazione su azze y da sistema di riferimento camera a sistema di riferimento robot
    P_c = np.array([P_c[2], -P_c[0], -P_c[1]], dtype=np.float64)
    return P_c


def transform_to_waist(    point_cam: np.ndarray,
    T_cam_to_waist: np.ndarray) -> np.ndarray:
       
    # Validazione degli input    
    point_cam = np.asarray(point_cam, dtype=np.float64).flatten()    
    if point_cam.shape != (3,):
        raise ValueError(            f"Il punto camera deve avere 3 coordinate, ricevute {point_cam.shape[0]}"
        )
    if T_cam_to_waist.shape != (4, 4):
        raise ValueError(            f"T_cam_to_waist deve essere 4x4, ricevuta forma {T_cam_to_waist.shape}"
        )
    # Conversione in coordinate omogenee: aggiungiamo 1 come quarta componente  
    point_cam = camera_point_trasformation(point_cam) #trasformazione da sistema di riferimento camera a sistema di riferimento robot     
    point_cam_hom = np.array([
        point_cam[0],  point_cam[1], point_cam[2], 1.0 ], dtype=np.float64)

    # Applicazione della trasformazione    
    point_waist_hom = T_cam_to_waist @ point_cam_hom
    # Estrazione delle coordinate 3D (le prime 3 componenti)    
    point_waist = point_waist_hom[:3]

    return point_waist #ritorna il punto in coordinate del waist


def inverse_transform(T: np.ndarray) -> np.ndarray:

    R = T[:3, :3]    
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)    
    T_inv[:3, :3] = R.T                # R trasposta = R inversa (per rotazioni)    
    T_inv[:3, 3] = -R.T @ t            # Traslazione inversa

    return T_inv