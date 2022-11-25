using UnityEngine;
using DeepLearning;
using System.Threading;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;

#if UNITY_EDITOR
using UnityEditor;
#endif

public abstract class NeuralAnimation : MonoBehaviour {

	public enum FPS {Thirty, Sixty}

	public PoseNetwork PoseNetwork;
// 
	public Actor Actor;
	// public MotionData file;


	public float AnimationTime {get; private set;}
	public float PostprocessingTime {get; private set;}
	public FPS Framerate = FPS.Thirty;


	protected abstract void ImportTestingSequence();
	protected abstract void Setup();

	protected abstract IEnumerator InitializeScene();

	protected abstract void Feed();
	protected abstract void Read();
	protected abstract void Postprocess();


	protected abstract void OnGUIDerived();
	protected abstract void OnRenderObjectDerived();



    void Start() {
		ImportTestingSequence();
		StartCoroutine(InitializeScene());
		Setup();
		PoseNetwork.CreateSession();

    }



    void LateUpdate() {
		Utility.SetFPS(Mathf.RoundToInt(GetFramerate()));
		if(PoseNetwork != null) {

			System.DateTime t1 = Utility.GetTimestamp();
			PoseNetwork.ResetPivot();  Feed();
			PoseNetwork.Predict(); 
			PoseNetwork.ResetPivot();  Read();
			AnimationTime = (float)Utility.GetElapsedTime(t1);
			Postprocess();
			
		}	

    }

    void OnGUI() {
		if(PoseNetwork != null) {
			OnGUIDerived();
		}
    }

	void OnRenderObject() {
		if(PoseNetwork != null) {
			if(Application.isPlaying) {
				OnRenderObjectDerived();
			}
		}
	}


	public float GetFramerate() {
		switch(Framerate) {
			case FPS.Thirty:
			return 30f;
			case FPS.Sixty:
			return 60f;
		}
		return 1f;
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(NeuralAnimation), true)]
	public class NeuralAnimation_Editor : Editor {

		public NeuralAnimation Target;

		void Awake() {
			Target = (NeuralAnimation)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			DrawDefaultInspector();

			EditorGUILayout.HelpBox("Animation: " + 1000f*Target.AnimationTime + "ms", MessageType.None);
			// EditorGUILayout.HelpBox("Postprocessing: " + 1000f*Target.PostprocessingTime + "ms", MessageType.None);

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}
	#endif

}
