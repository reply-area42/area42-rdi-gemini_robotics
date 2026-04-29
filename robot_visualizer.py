import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import os

def visualize_robot_on_meshcat(q_left, q_right, urdf_path="g1_dof_with_inspire_hands.urdf"):
    """
    Visualizza il robot su meshcat con gli angoli calcolati da solve_ik
    Parametri:
        q_left: angoli giunti braccio sinistro [7]
        q_right: angoli giunti braccio destro [7]
        urdf_path: path al file URDF del robot
    """
    # Carica il modello dal file URDF
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, urdf_dir)

    # Crea e inizializza il visualizzatore
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)   # apre il browser automaticamente
    viz.loadViewerModel()

    # Parte dalla postura neutra e sovrascrive i giunti delle braccia
    q = pin.neutral(model)
    n = min(14, model.nq)
    q[:7]  = q_left[:min(7, n)]
    q[7:n] = q_right[:min(7, n - 7)]

    viz.display(q)

    print("Robot visualizzato su meshcat!")
    print(f"Braccio sx: {q_left}")
    print(f"Braccio dx: {q_right}")
    print(f"URL: {viz.viewer.url()}")
    return viz


if __name__ == "__main__":
    # Diverse pose da visualizzare
    poses = [
        {
            "name": "Posizione Neutra",
            "q_left": np.array([0.0, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0]),
            "q_right": np.array([0.0, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0])
        },
        {
            "name": "Bracci Sollevati",
            "q_left": np.array([0.0, -0.2, 0.0, -0.8, 0.0, 0.3, 0.0]),
            "q_right": np.array([0.0, -0.2, 0.0, -0.8, 0.0, 0.3, 0.0])
        },
        {
            "name": "Braccio Sx Avanti",
            "q_left": np.array([0.5, -1.0, 0.3, -1.2, 0.2, 0.6, 0.0]),
            "q_right": np.array([0.0, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0])
        },
        {
            "name": "Braccio Dx Avanti",
            "q_left": np.array([0.0, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0]),
            "q_right": np.array([-0.5, -1.0, -0.3, -1.2, -0.2, 0.6, 0.0])
        },
        {
            "name": "Entrambi Sollevati Alto",
            "q_left": np.array([0.2, 0.2, -0.1, -0.5, 0.1, 0.2, 0.0]),
            "q_right": np.array([-0.2, 0.2, 0.1, -0.5, -0.1, 0.2, 0.0])
        },
        {
            "name": "Posizione Griping",
            "q_left": np.array([0.0, -1.2, 0.1, -1.0, 0.0, 0.8, 0.2]),
            "q_right": np.array([0.0, -1.2, -0.1, -1.0, 0.0, 0.8, -0.2])
        }
    ]

    try:
        # Visualizza prima la posizione neutra per aprire il browser
        viz = visualize_robot_on_meshcat(poses[0]["q_left"], poses[0]["q_right"])
        
        print("\n" + "="*50)
        print("Animazione posizioni del robot")
        print("="*50)
        print(f"Ogni posizione verrà visualizzata per 3 secondi\n")
        
        # Cicla attraverso tutte le pose
        while True:
            for i, pose in enumerate(poses):
                print(f"[{i+1}/{len(poses)}] {pose['name']}...")
                q_left = pose['q_left']
                q_right = pose['q_right']
                
                # Aggiorna la visualizzazione
                urdf_path = "g1_dof_with_inspire_hands.urdf"
                urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
                model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, urdf_dir)
                
                q = pin.neutral(model)
                n = min(14, model.nq)
                q[:7]  = q_left[:min(7, n)]
                q[7:n] = q_right[:min(7, n - 7)]
                
                viz.display(q)
                time.sleep(3)  # Mostra per 3 secondi
            
            print("\n" + "-"*50)
            print("Ricominciando dal inizio...")
            print("-"*50 + "\n")
            
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()