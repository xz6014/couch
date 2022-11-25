using UnityEngine;
using DeepLearning;
using UnityEditor;

public class Example_ONNX : MonoBehaviour {

    public PoseNetwork PoseNetwork;

    void Awake() {
        //Create a new inference session before running the network at each frame.
        PoseNetwork.CreateSession();
    }

    void OnDestroy() {
        //Close the session which disposes allocated memory.
        PoseNetwork.CloseSession();
    }

    void Update() {
        PoseNetwork.ResetPivot();

        //Give your inputs to the network. You can directly feed your inputs to the network without allocating the inputs array,
        //which is faster. If not enough or too many inputs are given to the network, it will throw warnings.
        float[] input = new float[PoseNetwork.Session.GetFeedSize()];
        for(int i=0; i<PoseNetwork.Session.GetFeedSize(); i++) {
            PoseNetwork.Feed(input[i]);
        }

        //Run the inference.
        PoseNetwork.Predict();
        PoseNetwork.ResetPivot();
        //Read your outputs from the network. You can directly read all outputs from the network without allocating the outputs array,
        //which is faster. If not enough or too many outputs are read from the network, it will throw warnings.
        float[] output = new float[PoseNetwork.Session.GetReadSize()];
        // Debug.Log(PoseNetwork.Session.GetReadSize());
        for(int i=0; i<PoseNetwork.Session.GetReadSize(); i++) {
            // Debug.Log(string.Format("{0} {1}", i, PoseNetwork.Session.Pivot));
            output[i] = PoseNetwork.Read();
        }

        // output.Print(false);
    }
}
#if UNITY_EDITOR
    [CustomEditor(typeof(PoseNetwork), true)]
    public class Example_ONNX_Editor : Editor {

        public Example_ONNX Target;

        void Awake() {
            Target = (Example_ONNX)target;
        }

        public override void OnInspectorGUI() {
            Undo.RecordObject(Target, Target.name);

            DrawDefaultInspector();

            if(GUI.changed) {
                EditorUtility.SetDirty(Target);
            }
        }

    }
#endif
