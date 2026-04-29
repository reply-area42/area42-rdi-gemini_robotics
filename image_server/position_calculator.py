import numpy as np
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """
    Parametri intrinseci della camera (modello pinhole).
    
    Attributi:
        fx: lunghezza focale lungo l'asse x (in pixel)
        fy: lunghezza focale lungo l'asse y (in pixel)
        cx: coordinata x del punto principale (centro ottico) in pixel
        cy: coordinata y del punto principale (centro ottico) in pixel
    """
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        """Validazione dei parametri intrinseci dopo l'inizializzazione."""
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"Le lunghezze focali devono essere positive: fx={self.fx}, fy={self.fy}"
            )
        if self.cx < 0 or self.cy < 0:
            raise ValueError(
                f"Il punto principale deve avere coordinate non negative: cx={self.cx}, cy={self.cy}"
            )

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
    point_cam_hom = np.array([
        point_cam[0],  point_cam[1], point_cam[2], 1.0 ], dtype=np.float64)

    # Applicazione della trasformazione    
    point_waist_hom = T_cam_to_waist @ point_cam_hom
    # Estrazione delle coordinate 3D (le prime 3 componenti)    
    point_waist = point_waist_hom[:3]

    return point_waist


def inverse_transform(T: np.ndarray) -> np.ndarray:

    R = T[:3, :3]    
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)    
    T_inv[:3, :3] = R.T                # R trasposta = R inversa (per rotazioni)    
    T_inv[:3, 3] = -R.T @ t            # Traslazione inversa

    return T_inv


if __name__ == "__main__":

    intrinsics = CameraIntrinsics(        
        fx=600.0,   # Sostituire con il valore reale (es. 386.738)        
        fy=600.0,   # Sostituire con il valore reale (es. 386.738)        
        cx=320.0,   # Sostituire con il valore reale (es. 321.281)        
        cy=240.0    # Sostituire con il valore reale (es. 238.221)
    )
    print(f"\n[Intrinseci camera]")    
    print(f"  fx={intrinsics.fx}, fy={intrinsics.fy}")
    print(f"  cx={intrinsics.cx}, cy={intrinsics.cy}")    
    print(f"\n  Matrice K:")    
    K = intrinsics.to_matrix()

    for i in range(3):        
        print(f"    [{K[i, 0]:8.2f}  {K[i, 1]:8.2f}  {K[i, 2]:8.2f}]")

    roll = 0.0                           # Sostituire con valore misurato [rad]    
    pitch = np.radians(-15.0)            # Sostituire con valore misurato [rad]    
    yaw = 0.0                            # Sostituire con valore misurato [rad]
      
    tx = 0.20    # Avanti rispetto alla vita [m] - Sostituire con valore misurato    
    ty = -0.05   # A sinistra rispetto alla vita [m] - Sostituire con valore misurato    
    tz = 0.30    # Sopra la vita [m] - Sostituire con valore misurato

    # Costruzione della matrice di rotazione e della trasformazione omogenea    
    R_cam_to_waist = build_rotation_matrix(roll, pitch, yaw)    
    t_cam_to_waist = np.array([tx, ty, tz])    
    T_cam_to_waist = build_T_cam_to_waist(R_cam_to_waist, t_cam_to_waist)
    

    # ESEMPIO DI UTILIZZO   
    # Coordinate pixel dell'oggetto rilevato nell'immagine    
    u_pixel = 400.0      # Colonna (leggermente a destra del centro)    
    v_pixel = 300.0      # Riga (leggermente sotto il centro)    
    depth_value = 1500.0  # Profondità in millimetri (1.5 metri)

    # --- Passo 1: Deproiezione nel frame camera ---    
    point_camera = deproject_pixel(u_pixel, v_pixel, depth_value, intrinsics)    

    # --- Passo 2: Trasformazione nel frame vita ---
    point_waist = transform_to_waist(point_camera, T_cam_to_waist)    
