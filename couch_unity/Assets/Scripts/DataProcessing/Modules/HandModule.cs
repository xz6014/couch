#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class HandModule : Module {

	public int RightWrist, LeftWrist;
	public Vector3 pos;
	public Vector3 vel;
	public Quaternion rot;
	public float DrawScale = 1f;
	public bool DrawHandTraj = true;
	public int Step = 10;
	public override ID GetID() {
		return ID.Hand;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		DetectSetup();
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		/*
		Actor actor = editor.GetActor();
		if(actor.Bones.Length == 0) {
			return;
		}
		if(actor.Bones[0].Transform == actor.transform) {
			return;
		}
		Transform bone = actor.Bones[0].Transform;
		Transform parent = bone.parent;
		bone.SetParent(null);
		Matrix4x4 root = GetRootTransformation(editor.GetCurrentFrame(), editor.Mirror);
		actor.transform.position = root.GetPosition();
		actor.transform.rotation = root.GetRotation();
		bone.SetParent(parent);
		bone.transform.localScale = Vector3.one;
		*/
	}

	protected override void DerivedDraw(MotionEditor editor) {
		UltiDraw.Begin();

		string[] BoneNames = new string[] {"m_avg_R_Wrist", "m_avg_L_Wrist"};
		Frame frame = editor.GetCurrentFrame();
		if(DrawHandTraj) {
			float start = Mathf.Clamp(frame.Timestamp-1, 0f, Data.GetTotalTime());
			float end = Mathf.Clamp(frame.Timestamp+1, 0f, Data.GetTotalTime());
			for(float j=start; j<=end; j+=Mathf.Max(Step, 1)/Data.Framerate) {
				Frame reference = Data.GetFrame(j);
				for(int i=0; i<2; i++) {
				UltiDraw.DrawSphere(GetPosition(reference, editor.Mirror, BoneNames[i]) ,Quaternion.identity, DrawScale*0.05f, UltiDraw.DarkRed);
				}
			}
		}
		UltiDraw.End();

	}

	protected override void DerivedInspector(MotionEditor editor) {
		// Topology = (TOPOLOGY)EditorGUILayout.EnumPopup("Topology", Topology);

		RightWrist = EditorGUILayout.Popup("Right Wrist", RightWrist, Data.Source.GetBoneNames());
		LeftWrist = EditorGUILayout.Popup("Left Wrist", LeftWrist, Data.Source.GetBoneNames());
		DrawHandTraj = EditorGUILayout.Toggle("Show Contacts", DrawHandTraj);
	}

	public void DetectSetup() {
		MotionData.Hierarchy.Bone rw = Data.Source.FindBoneContains("m_avg_R_Wrist");
		RightWrist = rw == null ? 0 : rw.Index;
		MotionData.Hierarchy.Bone lw = Data.Source.FindBoneContains("m_avg_L_Wrist");
		LeftWrist = lw == null ? 0 : lw.Index;
	}



	public Matrix4x4 GetTransformation(Frame frame, bool mirrored, string bone) {
		return Matrix4x4.TRS(GetPosition(frame, mirrored, bone), GetRotation(frame, mirrored, bone), Vector3.one);
	}

	public Vector3 GetPosition(Frame frame, bool mirrored, string bone) {
		
		pos = frame.GetBoneTransformation(bone, mirrored).GetPosition();
		return pos;

	}

	public Quaternion GetRotation(Frame frame, bool mirrored, string bone) {
		rot = frame.GetBoneTransformation(bone, mirrored).GetRotation();
		// rot[1] = frame.GetBoneTransformation(LeftWrist, mirrored).GetRotation();
		return rot;

	}

	public Vector3 GetVelocity(Frame frame, bool mirrored, float delta, string bone) {
		// vel = new Vector3[2]; 
		// vel[0] = frame.GetBoneVelocity(RightWrist, mirrored, delta);
		vel = frame.GetBoneVelocity(bone, mirrored, delta);
		return vel;
	}


	public Vector3[] GetHandToContactVec(Frame frame, bool mirrored, string bone) {
		Vector3[] diff = new Vector3[2];
		ContactModule m_contact = (ContactModule)Data.GetModule(Module.ID.Contact);
		Vector3 contact = m_contact.GetContact(frame, mirrored, bone);
		for(int i=0; i<2; i++){
			diff[i] = contact - frame.GetBoneTransformation(bone, mirrored).GetPosition();
		}
		return diff;

	}

	// public float GetRootSpeed(Frame frame, bool mirrored, float window, int step) {
	// 	float length = 0f;
	// 	int count = 0;
	// 	Vector3 prev = Vector3.zero;
	// 	while(true) {
	// 		float delta = step * count/Data.Framerate;
	// 		if(delta > window) {
	// 			break;
	// 		}
	// 		Vector3 pos = GetEstimatedRootPosition(frame, delta, mirrored);
	// 		pos.y = 0f;
	// 		if(count > 0) {
	// 			length += Vector3.Distance(prev, pos);
	// 		}
	// 		prev = pos;
	// 		count += 1;
	// 	}
	// 	return length / window;
	// }

	public Matrix4x4 GetEstimatedTransformation(Frame reference, float offset, bool mirrored, string bone) {
		return Matrix4x4.TRS(GetEstimatedPosition(reference, offset, mirrored, bone), GetEstimatedRotation(reference, offset, mirrored, bone), Vector3.one);
	}

	public Vector3 GetEstimatedPosition(Frame reference, float offset, bool mirrored, string bone) {
		float t = reference.Timestamp + offset;
		if(t < 0f || t > Data.GetTotalTime()) {
			float boundary = Mathf.Clamp(t, 0f, Data.GetTotalTime());
			float pivot = 2f*boundary - t;
			float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
			return 2f*GetPosition(Data.GetFrame(boundary), mirrored, bone) - GetPosition(Data.GetFrame(clamped), mirrored, bone);
		} else {
			return GetPosition(Data.GetFrame(t), mirrored, bone);
		}
	}

	public Quaternion GetEstimatedRotation(Frame reference, float offset, bool mirrored, string bone) {
		float t = reference.Timestamp + offset;
		if(t < 0f || t > Data.GetTotalTime()) {
			float boundary = Mathf.Clamp(t, 0f, Data.GetTotalTime());
			float pivot = 2f*boundary - t;
			float clamped = Mathf.Clamp(pivot, 0f, Data.GetTotalTime());
			return GetRotation(Data.GetFrame(clamped), mirrored, bone);
		} else {
			return GetRotation(Data.GetFrame(t), mirrored, bone);
		}
	}

	public Vector3 GetEstimatedVelocity(Frame reference, float offset, bool mirrored, float delta,  string bone) {
		return (GetEstimatedPosition(reference, offset + delta, mirrored, bone) - GetEstimatedPosition(reference, offset, mirrored, bone)) / delta;
	}

	
	// public Vector3 GetClosestTrajectory(Frame reference, float offset, bool mirrored, float delta,  int boneindex) {
		
	// 	return (GetEstimatedPosition(reference, offset + delta, mirrored, boneindex) - GetEstimatedPosition(reference, offset, mirrored, boneindex)) / delta;
	// }

	// public void ImportTestingSequences() {
	// 	// LoadData((MotionData)null);
	// 	string[] assets = AssetDatabase.FindAssets("t:MotionData", new string[1]{"Assets/MotionCapture/new_data_loaded2_noaug"});
	// 	// string[] assets = AssetDatabase.FindAssets("t:MotionData", new string[1]{"Assets/MotionCapture/rest_data_bvh_transformed"});

	// 	Files = new MotionData[assets.Length];
	// 	for(int i=0; i<assets.Length; i++) {
	// 		Files[i] = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(assets[i]), typeof(MotionData));
	// 	}
	// 	Debug.Log(string.Format("Successfully loaded {0} motion sequences", Files.Length.ToString()));
	// }
	/*
	public float GetTargetSpeed(Frame frame, bool mirrored, float window) {
		List<Vector3> positions = new List<Vector3>();
		int count = 0;
		while(true) {
			float delta = count/Data.Framerate;
			if(frame.Timestamp + delta > window) {
				break;
			}
			count += 1;
			Vector3 p = GetRootTransformation(frame, mirrored, delta).GetPosition();
			p.y = 0f;
			positions.Add(p);
		}
		if(positions.Count == 0) {
			Debug.Log("Oups! Something went wrong in computing target speed.");
			return 0f;
		} else {
			float speed = 0f;
			for(int i=1; i<positions.Count; i++) {
				speed += Vector3.Distance(positions[i-1], positions[i]);
			}
			return speed/window;
		};
	}
	*/

}
#endif
