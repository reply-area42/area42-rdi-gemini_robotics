
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import os

def visualize_robot_on_meshcat(q_left, q_right, urdf_path="assets/g1/g1_29dof.urdf", viz=None, model=None):
    """
    Visualizza il robot su meshcat con gli angoli calcolati da solve_ik
    Permette di riutilizzare la stessa istanza di viz e model tra più chiamate.
    Parametri:
        q_left: angoli giunti braccio sinistro [7]
        q_right: angoli giunti braccio destro [7]
        urdf_path: path al file URDF del robot
        viz: istanza MeshcatVisualizer opzionale
        model: modello Pinocchio opzionale
    """
    if model is None:
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, urdf_dir)
    else:
        collision_model = viz.collision_model
        visual_model = viz.visual_model

    if viz is None:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()

    # Parte dalla postura neutra e sovrascrive i giunti delle braccia
    q = pin.neutral(model)
    # Imposta solo i giunti delle braccia (indici 15-28)
    q[15:22] = q_left
    q[22:29] = q_right

    viz.display(q)

    print("Robot visualizzato su meshcat!")
    print(f"Braccio sx: {q_left}")
    print(f"Braccio dx: {q_right}")
    print(f"URL: {viz.viewer.url()}")
    return viz, model

def test_tutti_giunti(urdf_path="assets/g1/g1_29dof.urdf", amp=1.0, sleep_time=1.5):
    """
    Visualizza la variazione di ogni singolo giunto del robot (0-28) separatamente.
    Ogni giunto viene mosso uno alla volta, gli altri restano a zero.
    """
    # Nomi giunti secondo enum (0-28)
    joint_names = [
        # Left leg
        "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee", "LeftAnklePitch", "LeftAnkleRoll",
        # Right leg
        "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee", "RightAnklePitch", "RightAnkleRoll",
        # Waist
        "WaistYaw", "WaistRoll", "WaistPitch",
        # Left arm
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
        # Right arm
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow", "RightWristRoll", "RightWristPitch", "RightWristYaw",
        "kNotUsedJoint0", "kNotUsedJoint1", "kNotUsedJoint2", "kNotUsedJoint3", "kNotUsedJoint4", "kNotUsedJoint5"
    ]
    n_joints = 34
    q = np.zeros(n_joints)
    viz = None
    model = None
    print("Test movimento di tutti i giunti del robot:")
    for i, name in enumerate(joint_names):
        q_test = np.zeros(n_joints)
        q_test[i] = amp
        print(f"Muovo giunto {i}: {name}")
        if model is None:
            urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
            model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, urdf_dir)
        if viz is None:
            viz = MeshcatVisualizer(model, collision_model, visual_model)
            viz.initViewer(open=True)
            viz.loadViewerModel()
        q_full = pin.neutral(model)
        # Aggiorna tutti i giunti secondo la nuova mappatura
        q_full[:min(len(q_full), n_joints)] = q_test[:min(len(q_full), n_joints)]
        viz.display(q_full)
        while True:
            user_input = input("Premi 'a' per passare al prossimo giunto, oppure 'q' per uscire: ")
            if user_input.lower() == 'a':
                break
            elif user_input.lower() == 'q':
                print("Test interrotto dall'utente.")
                return

if __name__ == "__main__":
    # Lista di pose con descrizione
    # Mappatura giunti braccio (sinistra/destra):
    # [ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw]
    # Esempi di pose fisiologiche e realistiche:
    poses = [
        # (descrizione, q_left, q_right)
        ("Neutra (braccia lungo i fianchi)",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("A T (braccia orizzontali)",
         np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("Avanti (braccia tese in avanti)",
         np.array([-1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
         np.array([-1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("In alto (braccia sopra la testa)",
         np.array([1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
         np.array([1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    ]

    viz = None
    model = None
    try:
        # for desc, q_left, q_right in poses:
        #     print(f"Visualizzo: {desc}")
        #     # Puoi anche aggiungere la descrizione come commento qui:
        #     # {desc}
        #     viz, model = visualize_robot_on_meshcat(q_left, q_right, viz=viz, model=model)
        #     time.sleep(10)

        print("\n--- TEST TUTTI I GIUNTI DEL ROBOT ---")
        test_tutti_giunti(urdf_path="assets/g1/g1_29dof.urdf", sleep_time=10)
        print("\nVisualizzazione completata! Premi Ctrl+C per uscire")
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()

