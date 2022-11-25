#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;
using System.Linq;
public class ContactModule : Module {

	public float DrawScale = 1f;
	public bool ShowDebug = false;
	public bool ShowSensors = false;
	public bool ShowTolerances = false;
	public bool ShowDistances = false;
	public bool TrueMotionTrajectory = false;
	public bool CorrectedMotionTrajectory = false;
	public bool ShowContacts = false;
	public bool ContactTrajectories = false;
	public bool ShowClusteredContacts = false;
	public bool HandTrajectory = false;
	public bool ShowProcessedContacts = true;
	public bool ShowLocalPhase = true;

	//public bool ShowSkeletons = false;
	//public int SkeletonStep = 10;

	public bool EditMotion = true;
	public int Step = 10;
	public float CaptureFilter = 0.235f;
	public float EditFilter = 0.1f;
	public Sensor[] Sensors = new Sensor[0];
	public BakedContacts BakedContacts = null;

	public float PastTrajectoryWindow = 1f;
	public float FutureTrajectoryWindow = 1f;

	private float Window = 1f;

	private UltimateIK.Model IK;
	
	public Vector3[][] RegularHandContactVector;
	public Vector3[][] InverseHandContactVector;

	public override ID GetID() {
		return ID.Contact;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		if(EditMotion) {
			float delta = 1f/editor.TargetFramerate;
			Actor actor = editor.GetActor();
			IK = UltimateIK.BuildModel(IK, actor.Bones[0].Transform, GetObjectives(actor));
			// IK_LHand = UltimateIK.BuildModel(IK, Actor.FindTransform("m_avg_L_Shoulder"), GetObjectives(actor));
			// IK_RHand = UltimateIK.BuildModel(IK, Actor.FindTransform("m_avg_R_Shoulder"), GetObjectives(actor));

			if(IK.Objectives.Length == 0) {
				return;
			}
			IK.Iterations = 50;
			IK.Activation = UltimateIK.ACTIVATION.Constant;
			IK.RootTranslationY = true;
			IK.SetWeights(GetWeights());
			bool[] solvePositions = GetSolvePositions();
			bool[] solveRotations = GetSolveRotations();
			for(int i=0; i<IK.Objectives.Length; i++) {
				IK.Objectives[i].SolvePosition = solvePositions[i];
				IK.Objectives[i].SolveRotation = solveRotations[i];
			}
			
			Frame frame = editor.GetCurrentFrame();
			string [] bones = new string[1];
			// bones[0] = "m_avg_Pelvis";	
			// if (GetContacts(editor.GetPreviousFrame(), false, bones)[0]==1 &  GetContacts(frame, false, bones)[0]==0){
			// if (frame.Index==501){
			// 	Debug.Log(GetTargets(frame, editor.Mirror)[2].GetPosition().ToString("F3"));
			// }
			Frame relative = (frame.Timestamp - delta) < 0f ? Data.GetFrame(frame.Timestamp + delta) : Data.GetFrame(frame.Timestamp - delta);
			actor.WriteTransforms(relative.GetBoneTransformations(editor.Mirror), Data.Source.GetBoneNames());
			IK.Solve(GetTargets(relative, editor.Mirror));
			Matrix4x4[] relativePosture = actor.GetBoneTransformations();
			actor.WriteTransforms(frame.GetBoneTransformations(editor.Mirror), Data.Source.GetBoneNames());
			IK.Solve(GetTargets(frame, editor.Mirror));
			Matrix4x4[] framePosture = actor.GetBoneTransformations();
			
			for(int i=0; i<actor.Bones.Length; i++) {
				actor.Bones[i].Velocity = (frame.Timestamp - delta) < 0f ? 
				(relativePosture[i].GetPosition() - framePosture[i].GetPosition()) / delta:
				(framePosture[i].GetPosition() - relativePosture[i].GetPosition()) / delta;
			}
		}
	}

	public Sensor AddSensor() {
		return AddSensor(Data.Source.Bones[0].Name);
	}

	public Sensor AddSensor(string bone) {
		return AddSensor(bone, Vector3.zero, 0.1f, 0f, 0f);
	}

	public Sensor AddSensor(string bone, Vector3 offset, float threshold, float tolerance, float velocity) {
		return AddSensor(bone, offset, threshold, tolerance, velocity, Sensor.ID.Closest, Sensor.ID.None);
	}

	public Sensor AddSensor(string bone, Vector3 offset, float threshold, float tolerance, float velocity, Sensor.ID capture, Sensor.ID edit) {
		Sensor sensor = new Sensor(this, Data.Source.FindBone(bone).Index, offset, threshold, tolerance, velocity, capture, edit);
		ArrayExtensions.Add(ref Sensors, sensor);
		return sensor;
	}

	public void RemoveSensor(Sensor sensor) {
		if(!ArrayExtensions.Remove(ref Sensors, sensor)) {
			Debug.Log("Sensor could not be found in " + Data.GetName() + ".");
		}
	}

	public void Clear() {
		ArrayExtensions.Clear(ref Sensors);
	}

	public string[] GetNames() {
		string[] names = new string[Sensors.Length];
		for(int i=0; i<Sensors.Length; i++) {
			names[i] = Sensors[i].GetName();
		}
		return names;
	}

	public Sensor GetSensor(string bone) {
		return System.Array.Find(Sensors, x => x.GetName() == bone);
	}

	public float[] GetContacts(Frame frame, bool mirrored) {
		float[] contacts = new float[Sensors.Length];
		for(int i=0; i<Sensors.Length; i++) {
			contacts[i] = Sensors[i].GetContact(frame, mirrored);
		}
		return contacts;
	}

	public float[] GetContacts(Frame frame, bool mirrored, params string[] bones) {
		float[] contacts = new float[bones.Length];
		for(int i=0; i<bones.Length; i++) {
			Sensor sensor = GetSensor(bones[i]);
			if(sensor == null) {
				Debug.Log("Sensor for bone " + bones[i] + " could not be found.");
				contacts[i] = 0f;
			} else {
				contacts[i] = sensor.GetContact(frame, mirrored);
			}
		}
		return contacts;
	}


	public Vector3[] GetSensorOffsets(params string[] bones) {
		Vector3[] SensorOffsets = new Vector3[bones.Length];
		for(int i=0; i<bones.Length; i++) {
			SensorOffsets[i] = GetSensor(bones[i]).Offset;
		}
		return SensorOffsets;
	}

	// public void GetHandContactVector() {
	// 	RegularHandContactVector = new Vector3[Data.GetTotalFrames()][];
	// 	InverseHandContactVector = new Vector3[Data.GetTotalFrames()][];
	// 	var BoneNames = new List<string> {"m_avg_R_Wrist", "m_avg_L_Wrist"};


	// 	for(int i=0; i<RegularHandContactVector.Length; i++) {
	// 		RegularHandContactVector[i] = new Vector3[2];
	// 		InverseHandContactVector[i] = new Vector3[2];
	// 	}

	// 	for(int j=0; j<BoneNames.Count(); j++) {
	// 		Vector3[] RegularContacts = GetProcessedContacts(false, BoneNames[j]);
	// 		Vector3[] InverseContacts = GetProcessedContacts(true, BoneNames[j]);
	// 		for(int i=0; i<RegularHandContactVector.Length; i++) {
	// 			// Debug.Log(string.Format("{0} {1}", RegularHandContactVector.Length);
	// 			RegularHandContactVector[i][j] = GetCurrentHandTarget(Data.GetFrame(i), false, BoneNames[j]).GetPosition() - RegularContacts[i];
	// 			InverseHandContactVector[i][j] = GetCurrentHandTarget(Data.GetFrame(i), true, BoneNames[j]).GetPosition() - InverseContacts[i];

	// 		}
	// 	}
	// }

	// public Vector3

	public Matrix4x4[][] GetTargetsTrans(Frame frame, bool mirrored) {
		// List<Matrix4x4> targets = new List<Matrix4x4>();
		// float num_samples_f = 13;
		int num_samples = 13;
		// Debug.Log(num_samples);

		Matrix4x4[][] targets = new Matrix4x4[num_samples][];
		for(int i=0; i<targets.Length; i++) {
			targets[i] = new Matrix4x4[2];
		}
		for(int j=0; j<2; j++) {
			for(int i=0; i<targets.Length; i++) {
				targets[i][j] = Matrix4x4.identity;
			}
		}
		// Debug.Log(num_samples);
		float start = Mathf.Clamp(frame.Timestamp-PastTrajectoryWindow, 0f, Data.GetTotalTime());
	
		for(int j=0; j<num_samples; j++) {
			Frame reference = Data.GetFrame(start + Mathf.Max(Step, 1)/Data.Framerate * j);
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].GetName().Contains("Wrist")){

					// Debug.Log(i);
					// Vector3 previousPos =  Vector3.zero;
					// Vector3 previousCorrected = Vector3.zero;
					// Matrix4x4 bone = reference.GetBoneTransformation(Sensors[i].Bone, mirrored);
					targets[j][i-1]= Sensors[i].GetCorrectedTransformation(reference, mirrored);
					// previousPos = bone.GetPosition();
					// previousCorrected = corrected.GetPosition();
				}
				// targets.Add(corrected);
			}
		}
		return targets;
	}

	public Matrix4x4 GetCurrentHandTarget(Frame frame, bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		return sensor.GetCorrectedTransformation(frame, mirrored);
		
	}

	
	public Matrix4x4[] GetTargets(Frame frame, bool mirrored) {
		List<Matrix4x4> targets = new List<Matrix4x4>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				targets.Add(Sensors[i].GetCorrectedTransformation(frame, mirrored));
			}
		}
		return targets.ToArray();
	}

	public Transform[] GetObjectives(Actor actor) {
		List<Transform> objectives = new List<Transform>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				objectives.Add(actor.FindTransform(Data.Source.Bones[Sensors[i].Bone].Name));
			}
		}
		return objectives.ToArray();
	}

	public float[] GetWeights() {
		List<float> weights = new List<float>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				weights.Add(Sensors[i].Weight);
			}
		}
		return weights.ToArray();
	}

	public bool[] GetSolvePositions() {
		List<bool> values = new List<bool>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				values.Add(Sensors[i].SolvePosition);
			}
		}
		return values.ToArray();
	}

	public bool[] GetSolveRotations() {
		List<bool> values = new List<bool>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				values.Add(Sensors[i].SolveRotation);
			}
		}
		return values.ToArray();
	}


	public Vector3 GetClusteredContact(Frame frame, bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		return sensor.GetClusteredContact(frame,  mirrored);
	}
		
	public Vector3 GetContact(Frame frame, bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		return sensor.GetContactPoint(frame,  mirrored);
	}

	///////////////  IMPORTANT ///  Process and gettting Goal Points 
	public void GetProcessedContacts(bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		sensor.ProcessContactPoints(mirrored);
	}

	public Vector3 GetCorrectedGoalPoint(Frame frame, bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		return sensor.GetCorrectedGoalPoint(frame, mirrored);
	}


	/////////////////////////////////////////////////////////////////

	// Local Phases and that ....
	public float GetLocalPhases(Frame frame, bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		return sensor.GetLocalPhase(frame, mirrored);
	}

	public Vector2 GetLocalPhaseVectors(Frame frame, bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		return sensor.GetLocalPhaseVector(frame, mirrored);
	}
	public float GetFrequencies(Frame frame, bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		return sensor.GetFrequency(frame, mirrored);
	}
	public float GetAmplitudes(Frame frame, bool mirrored, string bone) {
		Sensor sensor = GetSensor(bone);
		return sensor.GetAmplitude(frame, mirrored);
	}



	public IEnumerator CaptureContacts(MotionEditor editor) {
		bool edit = EditMotion;
		EditMotion = false;
		Frame current = editor.GetCurrentFrame();
		for(int s=0; s<Sensors.Length; s++) {
			Sensors[s].RegularContacts = new float[Data.Frames.Length];
			Sensors[s].InverseContacts = new float[Data.Frames.Length];
			Sensors[s].RegularContactPoints = new Vector3[Data.Frames.Length];
			Sensors[s].InverseContactPoints = new Vector3[Data.Frames.Length];
			Sensors[s].RegularDistances = new Vector3[Data.Frames.Length];
			Sensors[s].InverseDistances = new Vector3[Data.Frames.Length];
		}
		System.DateTime time = Utility.GetTimestamp();
		//int count = 0;
		for(int i=0; i<Data.Frames.Length; i++) {
			//count += 1;
			Frame frame = Data.Frames[i];
			editor.LoadFrame(frame);
			for(int s=0; s<Sensors.Length; s++) {
				Sensors[s].CaptureContact(frame, editor);
			}
			//if(count > Step) {
			if(Utility.GetElapsedTime(time) > 0.2f) {
				time = Utility.GetTimestamp();
				//count = 0;
				yield return new WaitForSeconds(0f);
			}
			//}
		}
		// Clustering contacts
		for(int s=0; s<Sensors.Length; s++) {
			// if (Sensors[s].GetName().Contains("Ankle")){
				Sensors[s].ClusterContactPoints();
			// }
		}
		editor.LoadFrame(current);
		EditMotion = edit;
	}

	public void CaptureContactsNoCoroutine(MotionEditor editor) {
		bool edit = EditMotion;
		EditMotion = false;
		Frame current = editor.GetCurrentFrame();
		for(int s=0; s<Sensors.Length; s++) {
			Sensors[s].RegularContacts = new float[Data.Frames.Length];
			Sensors[s].InverseContacts = new float[Data.Frames.Length];
			Sensors[s].RegularContactPoints = new Vector3[Data.Frames.Length];
			Sensors[s].InverseContactPoints = new Vector3[Data.Frames.Length];
			Sensors[s].RegularDistances = new Vector3[Data.Frames.Length];
			Sensors[s].InverseDistances = new Vector3[Data.Frames.Length];
		}
		for(int i=0; i<Data.Frames.Length; i++) {
			Frame frame = Data.Frames[i];
			editor.LoadFrame(frame);
			for(int s=0; s<Sensors.Length; s++) {
				Sensors[s].CaptureContact(frame, editor);
			}
		}
		editor.LoadFrame(current);
		EditMotion = edit;
	}

	public void BakeContacts(MotionEditor editor) {
		if(BakedContacts == null) {
			return;
		}
		BakedContacts.Setup(GetNames(), Data.GetTotalFrames());
		for(int i=0; i<Data.Frames.Length; i++) {
			for(int s=0; s<Sensors.Length; s++) {
				if(Sensors[s].GetContact(Data.Frames[i], false) == 1f) {
					BakedContacts.BakeContact(s, Sensors[s].GetContactPoint(Data.Frames[i], false), Data.Frames[i], false);
				}
				if(Sensors[s].GetContact(Data.Frames[i], true) == 1f) {
					BakedContacts.BakeContact(s, Sensors[s].GetContactPoint(Data.Frames[i], true), Data.Frames[i], true);
				}
			}
		}
	}

	protected override void DerivedDraw(MotionEditor editor) {
		UltiDraw.Begin();
		
		Frame frame = editor.GetCurrentFrame();

		Color[] colors = UltiDraw.GetRainbowColors(Sensors.Length);

		if(ShowDebug) {
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].GetContact(frame, editor.Mirror) == 1f) {
					Vector3 contact = Sensors[i].GetContactPoint(frame, editor.Mirror);
					Vector3 corrected = Sensors[i].GetCorrectedContactPoint(frame, editor.Mirror);
					UltiDraw.DrawArrow(contact, corrected, 0.8f, 0.01f, DrawScale*0.025f, colors[i].Transparent(0.5f));
					UltiDraw.DrawSphere(contact, Quaternion.identity, DrawScale*0.025f, UltiDraw.Yellow);
					UltiDraw.DrawSphere(corrected, Quaternion.identity, DrawScale*0.05f, UltiDraw.Gold.Transparent(0.5f));
				}
			}
			for(int i=0; i<Sensors.Length; i++) {
				Matrix4x4 bone = frame.GetBoneTransformation(Sensors[i].Bone, editor.Mirror);
				Matrix4x4 corrected = Sensors[i].GetCorrectedTransformation(frame, editor.Mirror);
				UltiDraw.DrawCube(bone, DrawScale*0.025f, UltiDraw.DarkRed.Transparent(0.5f));
				UltiDraw.DrawLine(bone.GetPosition(), corrected.GetPosition(), colors[i].Transparent(0.5f));
				UltiDraw.DrawCube(corrected, DrawScale*0.025f, UltiDraw.DarkGreen.Transparent(0.5f));
			}
		}

		if(ShowSensors) {
			for(int i=0; i<Sensors.Length; i++) {
				Quaternion rot = editor.GetActor().GetBoneTransformation(Sensors[i].GetName()).GetRotation();
				Vector3 pos = editor.GetActor().GetBoneTransformation(Sensors[i].GetName()).GetPosition() + rot * Sensors[i].Offset;
				UltiDraw.DrawCube(pos, rot, DrawScale*0.025f, UltiDraw.Black);
				UltiDraw.DrawWireSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i].Transparent(0.25f));
				if(Sensors[i].GetContact(frame, editor.Mirror) == 1f) {
					UltiDraw.DrawSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i]);
				} else {
					UltiDraw.DrawSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i].Transparent(0.125f));
				}
			}
		}

		if(ShowTolerances) {
			for(int i=0; i<Sensors.Length; i++) {
				Quaternion rot = editor.GetActor().GetBoneTransformation(Sensors[i].GetName()).GetRotation();
				Vector3 pos = editor.GetActor().GetBoneTransformation(Sensors[i].GetName()).GetPosition() + rot * Sensors[i].Offset;
				UltiDraw.DrawWireSphere(pos, rot, 2f*(Sensors[i].Tolerance+Sensors[i].Threshold), UltiDraw.DarkGrey.Transparent(0.25f));
			}
		}

		if(ShowContacts) {
			// for(int i=0; i<Sensors.Length; i++) {
			// 	if(Sensors[i].Edit != Sensor.ID.None) {
			// 		for(float j=0f; j<=Data.GetTotalTime(); j+=Mathf.Max(Step, 1)/Data.Framerate) {
			// 			Frame reference = Data.GetFrame(j);
			// 			if(Sensors[i].GetContact(reference, editor.Mirror) == 1f) {
			// 				UltiDraw.DrawSphere(Sensors[i].GetContactPoint(reference, editor.Mirror), Quaternion.identity, DrawScale*0.025f, colors[i]);
			// 			}
			// 		}
			// 	}
			// }
			for(int i=1; i<Sensors.Length; i++) {
				// if (Sensors[i].GetName().Contains("Wrist")) {
					// Debug.Log(string.Format("{0} {1} {2}", Sensors[i].GetName(), frame.Index,Sensors[i].GetClusteredContact(frame, editor.Mirror).z));
				// }
				// if (Sensors[i].GetName().Contains("Ankle")) {
					if(Sensors[i].GetContactPoint(frame, editor.Mirror) != Vector3.zero) {
						UltiDraw.DrawSphere(Sensors[i].GetContactPoint(frame, editor.Mirror), Quaternion.identity, DrawScale*0.15f, colors[i].Transparent(0.8f));
					}
				// }
			}
		}

		if(ShowClusteredContacts) {

			for(int i=0; i<Sensors.Length; i++) {
				// if (Sensors[i].GetName().Contains("Wrist")) {
					// Debug.Log(string.Format("{0} {1} {2}", Sensors[i].GetName(), frame.Index,Sensors[i].GetClusteredContact(frame, editor.Mirror).z));
				// }
				if (Sensors[i].GetName().Contains("Ankle")) {
					if(Sensors[i].GetClusteredContact(frame, editor.Mirror) != Vector3.zero) {
						UltiDraw.DrawSphere(Sensors[i].GetClusteredContact(frame, editor.Mirror), Quaternion.identity, DrawScale*0.1f, colors[i].Transparent(0.8f));
					}
				}
			}
		}

		if(ShowProcessedContacts) {
			for(int i=1; i<5; i++) {
				// if (Sensors[i].GetName().Contains("Wrist")) {
					// Debug.Log(string.Format("{0} {1} {2}", Sensors[i].GetName(), frame.Index,Sensors[i].GetClusteredContact(frame, editor.Mirror).z));
				// }
				Sensors[i].ProcessContactPoints(editor.Mirror);
				UltiDraw.DrawSphere(Sensors[i].GetCorrectedGoalPoint(frame, editor.Mirror), Quaternion.identity, DrawScale*0.2f, colors[i].Transparent(0.8f));
			}
		}
		/*
		if(ShowSkeletons) {
			UltiDraw.End();
			float start = Mathf.Clamp(frame.Timestamp-Window, 0f, Data.GetTotalTime());
			float end = Mathf.Clamp(frame.Timestamp+Window, 0f, Data.GetTotalTime());
			float inc = Mathf.Max(SkeletonStep, 1)/Data.Framerate;
			for(float j=start; j<=end; j+=inc) {
				Frame reference = Data.GetFrame(j);
				float weight = (j-start+inc) / (end-start+inc);
				editor.GetActor().Sketch(reference.GetBoneTransformations(editor.GetActor().GetBoneNames(), editor.Mirror), Color.Lerp(UltiDraw.Cyan, UltiDraw.Magenta, weight).Transparent(weight));
			}
			UltiDraw.Begin();
		}
		*/

		if(TrueMotionTrajectory || CorrectedMotionTrajectory) {
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].Edit != Sensor.ID.None) {
					Vector3 previousPos = Vector3.zero;
					Vector3 previousCorrected = Vector3.zero;
					float start = Mathf.Clamp(frame.Timestamp-PastTrajectoryWindow, 0f, Data.GetTotalTime());
					float end = Mathf.Clamp(frame.Timestamp+FutureTrajectoryWindow, 0f, Data.GetTotalTime());
					for(float j=start; j<=end; j+=Mathf.Max(Step, 1)/Data.Framerate) {
						Frame reference = Data.GetFrame(j);
						Matrix4x4 bone = reference.GetBoneTransformation(Sensors[i].Bone, editor.Mirror);
						Matrix4x4 corrected = Sensors[i].GetCorrectedTransformation(reference, editor.Mirror);
						
						if(j > start) {
							if(TrueMotionTrajectory) {
								UltiDraw.DrawArrow(previousPos, bone.GetPosition(), 0.8f, DrawScale*0.005f, DrawScale*0.025f, UltiDraw.DarkRed.Lighten(0.5f).Transparent(0.5f));
							}
							if(CorrectedMotionTrajectory) {
								UltiDraw.DrawArrow(previousCorrected, corrected.GetPosition(), 0.8f, DrawScale*0.005f, DrawScale*0.025f, UltiDraw.DarkGreen.Lighten(0.5f).Transparent(0.5f));
							}
							//UltiDraw.DrawLine(previousPos, bone.GetPosition(), UltiDraw.DarkRed.Transparent(0.5f));
							//UltiDraw.DrawLine(previousCorrected, corrected.GetPosition(), UltiDraw.DarkGreen.Transparent(0.5f));
						}
						previousPos = bone.GetPosition();
						previousCorrected = corrected.GetPosition();
						if(TrueMotionTrajectory) {
							UltiDraw.DrawCube(bone, DrawScale*0.025f, UltiDraw.DarkRed.Transparent(0.5f));
						}
						//UltiDraw.DrawLine(bone.GetPosition(), corrected.GetPosition(), colors[i].Transparent(0.5f));
						if(CorrectedMotionTrajectory) {
							UltiDraw.DrawCube(corrected, DrawScale*0.025f, UltiDraw.DarkGreen);
						}
					}
				}
			}
		}

		if(ContactTrajectories) {
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].Edit != Sensor.ID.None) {
					float start = Mathf.Clamp(frame.Timestamp-Window, 0f, Data.GetTotalTime());
					float end = Mathf.Clamp(frame.Timestamp+Window, 0f, Data.GetTotalTime());
					for(float j=0f; j<=Data.GetTotalTime(); j+=Mathf.Max(Step, 1)/Data.Framerate) {
						Frame reference = Data.GetFrame(j);
						if(Sensors[i].GetContact(reference, editor.Mirror) == 1f) {
							Vector3 contact = Sensors[i].GetContactPoint(reference, editor.Mirror);
							Vector3 corrected = Sensors[i].GetCorrectedContactPoint(reference, editor.Mirror);
							UltiDraw.DrawArrow(contact, corrected, 0.8f, Vector3.Distance(contact, corrected)*DrawScale*0.025f, Vector3.Distance(contact, corrected)*DrawScale*0.1f, colors[i].Lighten(0.5f).Transparent(0.5f));
							UltiDraw.DrawSphere(contact, Quaternion.identity, DrawScale*0.0125f, colors[i].Transparent(0.5f));
							UltiDraw.DrawSphere(corrected, Quaternion.identity, DrawScale*0.05f, colors[i]);
						}
					}
				}
			}
		}

		if(HandTrajectory) {
			// Frame reference = Data.GetFrame(frame.Timestamp);
			Matrix4x4[][] corrected = GetTargetsTrans(frame, editor.Mirror);
			for(int i=6; i<corrected.Length; i++) {
				for(int j=0; j<corrected[0].Length; j++) {
					UltiDraw.DrawSphere(corrected[i][j].GetPosition(), Quaternion.identity, DrawScale*0.05f, colors[j]);
				}
			}		

		}

		if(ShowLocalPhase) {
			for(int i=0; i<Sensors.Length; i++) {
				if (i > 0) {
					if (Sensors[i].GetLocalPhaseVector(frame, editor.Mirror)[1] > 0.75f){
					// if (Sensors[i].GetLocalPhaseVector(frame, editor.Mirror)[1] > 0.85f | Sensors[i].GetLocalPhaseVector(frame, editor.Mirror)[1] < -0.85f){
						UltiDraw.DrawSphere(Sensors[i].GetCorrectedTransformation(frame, editor.Mirror).GetPosition(), Quaternion.identity, DrawScale*0.2f, colors[i].Transparent(0.8f));
					}
				}
			}
		}

		UltiDraw.End();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		if(Utility.GUIButton("Capture Contacts", UltiDraw.DarkGrey, UltiDraw.White)) {
			EditorCoroutines.StartCoroutine(CaptureContacts(editor), this);
		}
		EditorGUILayout.BeginHorizontal();
		BakedContacts = (BakedContacts)EditorGUILayout.ObjectField(BakedContacts, typeof(BakedContacts), true);
		EditorGUI.BeginDisabledGroup(BakedContacts == null || editor.Mirror);
		if(Utility.GUIButton("Bake", UltiDraw.DarkGrey, UltiDraw.White)) {
			BakeContacts(editor);
		}
		EditorGUI.EndDisabledGroup();
		EditorGUILayout.EndHorizontal();
		DrawScale = EditorGUILayout.FloatField("Draw Scale", DrawScale);
		EditMotion = EditorGUILayout.Toggle("Edit Motion", EditMotion);
		ShowDebug = EditorGUILayout.Toggle("Show Debug", ShowDebug);
		ShowSensors = EditorGUILayout.Toggle("Show Sensors", ShowSensors);
		ShowTolerances = EditorGUILayout.Toggle("Show Tolerances", ShowTolerances);
		ShowDistances = EditorGUILayout.Toggle("Show Distances", ShowDistances);
		TrueMotionTrajectory = EditorGUILayout.Toggle("True Motion Trajectory", TrueMotionTrajectory);
		CorrectedMotionTrajectory = EditorGUILayout.Toggle("Corrected Motion Trajectory", CorrectedMotionTrajectory);
		PastTrajectoryWindow = EditorGUILayout.FloatField("Past Trajectory Window", PastTrajectoryWindow);
		FutureTrajectoryWindow = EditorGUILayout.FloatField("Future Trajectory Window" , FutureTrajectoryWindow);
		//ShowSkeletons = EditorGUILayout.Toggle("Show Skeletons", ShowSkeletons);
		//SkeletonStep = EditorGUILayout.IntField("Skeleton Step", SkeletonStep);
		ShowContacts = EditorGUILayout.Toggle("Show Contacts", ShowContacts);
		ShowClusteredContacts = EditorGUILayout.Toggle("Show Clustered Contacts", ShowClusteredContacts);
		ContactTrajectories = EditorGUILayout.Toggle("Contact Trajectories", ContactTrajectories);
		HandTrajectory = EditorGUILayout.Toggle("Hand Trajectory", HandTrajectory);
		ShowProcessedContacts = EditorGUILayout.Toggle("Show Processed Contacts", ShowProcessedContacts);
		ShowLocalPhase = EditorGUILayout.Toggle("Show Local Phase", ShowLocalPhase);


		Step = EditorGUILayout.IntField("Step", Step);
		CaptureFilter = EditorGUILayout.Slider("Capture Filter", CaptureFilter, 0f, 1f);
		EditFilter = EditorGUILayout.Slider("Edit Filter", EditFilter, 0f, 1f);
		for(int i=0; i<Sensors.Length; i++) {
			EditorGUILayout.BeginHorizontal();
			Sensors[i].Inspector(editor);
			EditorGUILayout.BeginVertical();
			if(Utility.GUIButton("-", UltiDraw.DarkRed, UltiDraw.White, 28f, 18f)) {
				RemoveSensor(Sensors[i]);
			}
			EditorGUILayout.EndVertical();
			EditorGUILayout.EndHorizontal();
		}
		// if(!ArrayExtensions.Remove(ref Sensors, 'sensor'))
		// 	AddSensor("m_avg_Pelvis");
		// 	AddSensor("m_avg_L_Wrist");
		// 	AddSensor("m_avg_R_Wrist");
		// 	AddSensor("m_avg_L_Ankle");
		// 	AddSensor("m_avg_R_Ankle");

		if(Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White)) {
			AddSensor();
		}

		// Initialising sensor setting foor sitting
		if ((System.Array.Find(Sensors, x => x.GetName() == "m_avg_Pelvis"))== null){
			AddSensor("m_avg_Pelvis");
			GetSensor("m_avg_Pelvis").Capture = Sensor.ID.RayTopDown;
			GetSensor("m_avg_Pelvis").Edit = Sensor.ID.RayTopDown;
			GetSensor("m_avg_Pelvis").Threshold = 0.2f;
			GetSensor("m_avg_Pelvis").Tolerance = 0.5f;
		}
		// if ((System.Array.Find(Sensors, x => x.GetName() == "m_avg_R_Hand"))== null){
		// 	AddSensor("m_avg_R_Hand");
		// 	GetSensor("m_avg_R_Hand").Capture = Sensor.ID.Closest;
		// 	GetSensor("m_avg_R_Hand").Edit = Sensor.ID.Closest;



		// 	GetSensor("m_avg_R_Hand").Threshold = 0.05f;
		// 	GetSensor("m_avg_R_Hand").Tolerance = 0.3f;
		// 	GetSensor("m_avg_R_Hand").SolveDistance = false;


		// }	
		// if ((System.Array.Find(Sensors, x => x.GetName() == "m_avg_L_Hand"))== null){
		// 	AddSensor("m_avg_L_Hand");
		// 	GetSensor("m_avg_L_Hand").Capture = Sensor.ID.Closest;
		// 	GetSensor("m_avg_L_Hand").Edit = Sensor.ID.Closest;


		// 	GetSensor("m_avg_L_Hand").Threshold = 0.05f;
		// 	GetSensor("m_avg_L_Hand").Tolerance = 0.3f;
		// 	GetSensor("m_avg_L_Hand").SolveDistance = false;


		// }	

		if ((System.Array.Find(Sensors, x => x.GetName() == "m_avg_R_Wrist"))== null){
			AddSensor("m_avg_R_Wrist");
			GetSensor("m_avg_R_Wrist").Capture = Sensor.ID.Closest;
			GetSensor("m_avg_R_Wrist").Edit = Sensor.ID.Closest;

			GetSensor("m_avg_R_Wrist").Offset[0] = -0.07f;
			GetSensor("m_avg_R_Wrist").Offset[1] = -0.03f;


			GetSensor("m_avg_R_Wrist").Threshold = 0.05f;
			GetSensor("m_avg_R_Wrist").Tolerance = 0.3f;
			GetSensor("m_avg_R_Wrist").SolveDistance = false;


		}	
		if ((System.Array.Find(Sensors, x => x.GetName() == "m_avg_L_Wrist"))== null){
			AddSensor("m_avg_L_Wrist");
			GetSensor("m_avg_L_Wrist").Capture = Sensor.ID.Closest;
			GetSensor("m_avg_L_Wrist").Edit = Sensor.ID.Closest;

			GetSensor("m_avg_L_Wrist").Offset[0] = 0.07f;
			GetSensor("m_avg_L_Wrist").Offset[1] = -0.03f;

			GetSensor("m_avg_L_Wrist").Threshold = 0.05f;
			GetSensor("m_avg_L_Wrist").Tolerance = 0.3f;
			GetSensor("m_avg_L_Wrist").SolveDistance = false;


		}	

		if ((System.Array.Find(Sensors, x => x.GetName() == "m_avg_R_Ankle"))== null){
			AddSensor("m_avg_R_Ankle");
			GetSensor("m_avg_R_Ankle").Threshold = 0.07f;

			GetSensor("m_avg_R_Ankle").Offset[1] = -0.05f;
			GetSensor("m_avg_R_Ankle").Offset[2] = -0f;
			GetSensor("m_avg_R_Ankle").Velocity = 2f;

			GetSensor("m_avg_R_Ankle").Edit = Sensor.ID.Closest;

		}
		if ((System.Array.Find(Sensors, x => x.GetName() == "m_avg_L_Ankle"))== null){
			AddSensor("m_avg_L_Ankle");
			GetSensor("m_avg_L_Ankle").Threshold = 0.07f;
			GetSensor("m_avg_L_Ankle").Offset[1] = -0.05f;
			GetSensor("m_avg_L_Ankle").Offset[2] = -0f;
			GetSensor("m_avg_L_Ankle").Velocity = 2f;

			GetSensor("m_avg_L_Ankle").Edit = Sensor.ID.Closest;

		}		

	}


	[System.Serializable]
	public class Sensor {
		public enum ID {
			None, 
			Closest, 
			RayTopDown, RayCenterDown, RayBottomUp, RayCenterUp, 
			SphereTopDown, SphereCenterDown, SphereBottomUp, SphereCenterUp, 
			RayXPositive, RayXNegative, RayYPositive, RayYNegative, RayZPositive, RayZNegative,
			Identity
			};
		public ContactModule Module = null;
		public int Bone = 0;
		public Vector3 Offset = Vector3.zero;
		public float Threshold = 0.1f;
		public float Tolerance = 0f;
		public float Velocity = 0f;
		public bool SolvePosition = true;
		public bool SolveRotation = true;
		public bool SolveDistance = true;
		public LayerMask Mask = -1;
		public ID Capture = ID.Closest;
		public ID Edit = ID.None;
		public float Weight = 1f;

		public float[] RegularContacts, InverseContacts = new float[0];
		public Vector3[] RegularContactPoints, InverseContactPoints = new Vector3[0];
		public Vector3[] RegularDistances, InverseDistances = new Vector3[0];

		public Vector3[] ClusteredInverseContactPoints, ClusteredRegularContactPoints = new Vector3[0];
		
		public List<Vector3> reg_avg_contact_point;
		public List<Vector3> inv_avg_contact_point;

		public List<Vector3> reg_avg_goal_point;
		public List<Vector3> inv_avg_goal_point;

		public int reg_additional_key_frame;
		public int inv_additional_key_frame;


		public float[] RegularLocalPhase;
		public float[] InverseLocalPhase;

		
		//// Seb's feature
		public float[] RegularAmplitude;
		public float[] InverseAmplitude;

		public float[] RegularFrequency;
		public float[] InverseFrequency;

		public Vector2[] RegularLocalPhaseVector;
		public Vector2[] InverseLocalPhaseVector;

		// public Vector3[] CorrectedRegularGoalPoints;
		// public Vector3[] CorrectedInverseGoalPoints;
		// public Vector3[] ClusteredRegularGoalPoints;
		// public Vector3[] ClusteredInverseGoalPoints;

		public Vector3[] CorrectedRegularGoalPoints;
		public Vector3[] CorrectedInverseGoalPoints;

		public Vector3[] ClusteredRegularGoalPoints;
		public Vector3[] ClusteredInverseGoalPoints;

		
		public void SetAdditionalKeyFrame(int RegFrameIndex, int InvFrameIndex) {
			reg_additional_key_frame = RegFrameIndex;
			inv_additional_key_frame = InvFrameIndex;
			// Debug.Log(1);

			RegularLocalPhase = new float [Module.Data.Frames.Length];
			InverseLocalPhase = new float [Module.Data.Frames.Length];
			// if (RegFrameIndex != 0){
			// 	RegularContacts[RegFrameIndex-1] = 0f;
			// 	RegularContacts[RegFrameIndex] = 1f;
			// 	RegularContacts[RegFrameIndex+1] = 1f;
			// }
			// if (InvFrameIndex != 0){	
			// 	InverseContacts[InvFrameIndex-1] = 0f;
			// 	InverseContacts[InvFrameIndex] = 1f;
			// 	InverseContacts[InvFrameIndex+1] = 1f;
			// }
			// Debug.Log(FrameIndex);


			
		}

		public void LoadFrequencyLocalPhase(Vector2[] phases_reg, Vector2[] phases_inv, float[] frequencies_reg, float[] frequencies_inv) {

			RegularFrequency = frequencies_reg;
			InverseFrequency = frequencies_inv;



			RegularLocalPhaseVector = new Vector2[Module.Data.Frames.Length];
			InverseLocalPhaseVector = new Vector2[Module.Data.Frames.Length];

			RegularAmplitude = new float [Module.Data.Frames.Length];
			InverseAmplitude = new float [Module.Data.Frames.Length];
			
			for (int i=0; i<Module.Data.Frames.Length; i++){

				// Phase Vectors are normalized
				RegularLocalPhaseVector[i] = phases_reg[i].normalized;
				InverseLocalPhaseVector[i] = phases_inv[i].normalized;

				// Amplitudes are the magnitudes
				RegularAmplitude[i] = phases_reg[i].magnitude;
				InverseAmplitude[i] = phases_inv[i].magnitude;

			}

			
		}


		public Sensor(ContactModule module, int bone, Vector3 offset, float threshold, float tolerance, float velocity, ID capture, ID edit) {
			Module = module;
			Bone = bone;
			Offset = offset;
			Threshold = threshold;
			Tolerance = tolerance;
			Velocity = velocity;
			Capture = capture;
			Edit = edit;
			RegularContacts = new float[Module.Data.Frames.Length];
			InverseContacts = new float[Module.Data.Frames.Length];
			RegularContactPoints = new Vector3[Module.Data.Frames.Length];
			InverseContactPoints = new Vector3[Module.Data.Frames.Length];
			RegularDistances = new Vector3[Module.Data.Frames.Length];
			InverseDistances = new Vector3[Module.Data.Frames.Length];

			// clustered contact points, where contact point of consecutive frames are collected and is averaged 
			ClusteredRegularContactPoints = new Vector3[Module.Data.Frames.Length];
			ClusteredInverseContactPoints = new Vector3[Module.Data.Frames.Length];


			// RegularLocalPhase = new float [Module.Data.Frames.Length];
			// InverseLocalPhase = new float [Module.Data.Frames.Length];
		}
		

		public string GetName() {
			return Module.Data.Source.Bones[Bone].Name;
		}

		public int GetIndex() {
			return System.Array.FindIndex(Module.Sensors, x => x==this);
		}

		public Vector3 GetPivot(Frame frame, bool mirrored) {
			return Offset.GetRelativePositionFrom(frame.GetBoneTransformation(Bone, mirrored));
		}

		public float GetContact(Frame frame, bool mirrored) {
			return mirrored ? InverseContacts[frame.Index-1] : RegularContacts[frame.Index-1];
		}

		public Vector3 GetClusteredContact(Frame frame, bool mirrored) {
			return mirrored ? ClusteredInverseContactPoints[frame.Index-1] : ClusteredRegularContactPoints[frame.Index-1];
		}

		public Vector3 GetContactDistance(Frame frame, bool mirrored) {
			return mirrored ? InverseDistances[frame.Index-1] : RegularDistances[frame.Index-1];
		}

		public Vector3 GetContactPoint(Frame frame, bool mirrored) {
			return mirrored ? InverseContactPoints[frame.Index-1] : RegularContactPoints[frame.Index-1];
		}

		public Vector3 GetCorrectedGoalPoint(Frame frame, bool mirrored) {
			return mirrored ? CorrectedInverseGoalPoints[frame.Index-1] : CorrectedRegularGoalPoints[frame.Index-1];
		}
		public Vector3 GetClusteredGoalPoint(Frame frame, bool mirrored) {
			return mirrored ? ClusteredInverseGoalPoints[frame.Index-1] : ClusteredRegularGoalPoints[frame.Index-1];
		}



		public Vector3 GetCorrectedContactDistance(Frame frame, bool mirrored) {
			Matrix4x4 bone = frame.GetBoneTransformation(Bone, mirrored);
			if(SolveDistance) {
				return GetCorrectedContactPoint(frame, mirrored) - GetContactDistance(frame, mirrored) - bone.GetPosition();
			} else {
				return GetCorrectedContactPoint(frame, mirrored) - bone.GetRotation()*Offset - bone.GetPosition();
			}
		}

		public Vector3 GetCorrectedContactPoint(Frame frame, bool mirrored) {
			Collider collider = null;
			Vector3 point = DetectCollision(frame, mirrored, Edit, GetPivot(frame, mirrored), Tolerance+Threshold, out collider);
			if(collider != null) {
				Interaction annotated = collider.GetComponentInParent<Interaction>();
				if(annotated != null) {
					if(annotated.ContainsContact(GetName())) {
						point = annotated.GetContact(GetName(), frame, mirrored).GetPosition();
					}
					// Transform t = annotated.GetContactTransform(GetName());
					// if(t != null) {
					// 	if(mirrored) {
					// 		point = t.parent.position + t.parent.rotation * Vector3.Scale(t.parent.lossyScale.GetMirror(Module.Data.MirrorAxis), t.localPosition);
					// 	} else {
					// 		point = t.position;
					// 	}
					// }
				}
				BakedContacts baked = collider.GetComponentInParent<BakedContacts>();
				if(baked != null) {
					return baked.GetContactPoint(GetName(), frame, mirrored);
				}
			}
			return point;
		}

		public Matrix4x4 GetCorrectedTransformation(Frame frame, bool mirrored) {
			Matrix4x4 bone = frame.GetBoneTransformation(Bone, mirrored);
			if(Edit == ID.None) {
				return bone;
			}
			if(Edit == ID.Identity) {
				return bone;
			}
			if(GetContact(frame, mirrored) == 1f) {
				//Gaussian smoothing filter along contact points
				int width = Mathf.RoundToInt(Module.EditFilter * Module.Data.Framerate);
				bool[] contacts = new bool[2*width + 1];
				Vector3[] distances = new Vector3[2*width + 1];
				contacts[width] = true;
				distances[width] = GetCorrectedContactDistance(frame, mirrored);
				for(int i=1; i<=width; i++) {
					int left = frame.Index - i;
					int right = frame.Index + i;
					if(left > 1 && right <= Module.Data.GetTotalFrames()) {
						if(GetContact(Module.Data.GetFrame(left), mirrored) == 1f && GetContact(Module.Data.GetFrame(right), mirrored) == 1f) {
							contacts[width-i] = true;
							contacts[width+i] = true;
							distances[width-i] = GetCorrectedContactDistance(Module.Data.GetFrame(left), mirrored);
							distances[width+i] = GetCorrectedContactDistance(Module.Data.GetFrame(right), mirrored);
						} else {
							break;
						}
					} else {
						break;
					}
				}
				return Matrix4x4.TRS(bone.GetPosition() + Utility.FilterGaussian(distances, contacts), bone.GetRotation(), Vector3.one);
			} else {
				//Interpolation between ground truth and contact points
				float min = Mathf.Clamp(frame.Timestamp-Module.Window, 0f, Module.Data.GetTotalTime());
				float max = Mathf.Clamp(frame.Timestamp+Module.Window, 0f, Module.Data.GetTotalTime());
				Frame start = null;
				Frame end = null;
				for(float j=frame.Timestamp; j>=min; j-=1f/Module.Data.Framerate) {
					Frame reference = Module.Data.GetFrame(j);
					if(GetContact(reference, mirrored) == 1f) {
						start = reference;
						break;
					}
				}
				for(float j=frame.Timestamp; j<=max; j+=1f/Module.Data.Framerate) {
					Frame reference = Module.Data.GetFrame(j);
					if(GetContact(reference, mirrored) == 1f) {
						end = reference;
						break;
					}
				}
				if(start != null && end == null) {
					float weight = 1f - (frame.Timestamp - start.Timestamp) / (frame.Timestamp - min);
					return Matrix4x4.TRS(bone.GetPosition() + weight*GetCorrectedContactDistance(start, mirrored), bone.GetRotation(), Vector3.one);
				}
				if(start == null && end != null) {
					float weight = 1f - (end.Timestamp - frame.Timestamp) / (max - frame.Timestamp);
					return Matrix4x4.TRS(bone.GetPosition() + weight*GetCorrectedContactDistance(end, mirrored), bone.GetRotation(), Vector3.one);
				}
				if(start != null && end != null) {
					float weight = (frame.Timestamp - start.Timestamp) / (end.Timestamp - start.Timestamp);
					return Matrix4x4.TRS(
						bone.GetPosition() + Vector3.Lerp(GetCorrectedContactDistance(start, mirrored), GetCorrectedContactDistance(end, mirrored), weight), 
						bone.GetRotation(), 
						Vector3.one
					);
				}
				return bone;
			}
		}

		// Local Phases and that ....
		public float GetLocalPhase(Frame frame, bool mirrored) {
			return mirrored ? InverseLocalPhase[frame.Index-1] : RegularLocalPhase[frame.Index-1];
		}

		public Vector2 GetLocalPhaseVector(Frame frame, bool mirrored) {
			return mirrored ? InverseLocalPhaseVector[frame.Index-1] : RegularLocalPhaseVector[frame.Index-1];
		}		
		public float GetFrequency(Frame frame, bool mirrored) {
			return mirrored ? InverseFrequency[frame.Index-1] : RegularFrequency[frame.Index-1];
		}

		public float GetAmplitude(Frame frame, bool mirrored) {
			return mirrored ? InverseAmplitude[frame.Index-1] : RegularAmplitude[frame.Index-1];
		}



		public Vector3 DetectCollision(Frame frame, bool mirrored, Sensor.ID mode, Vector3 pivot, float radius, out Collider collider) {
			if(mode == ID.Closest) {
				return Utility.GetClosestPointOverlapSphere(pivot, radius, Mask, out collider);
			}

			if(mode == ID.RayTopDown) {
				RaycastHit info;
				bool hit = Physics.Raycast(pivot + new Vector3(0f, radius, 0f), Vector3.down, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayCenterDown) {
				RaycastHit info;
				bool hit = Physics.Raycast(pivot, Vector3.down, out info, radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayBottomUp) {
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - new Vector3(0f, radius, 0f), Vector3.up, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayCenterUp) {
				RaycastHit info;
				bool hit = Physics.Raycast(pivot, Vector3.up, out info, radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.SphereTopDown) {
				RaycastHit info;
				bool hit = Physics.SphereCast(pivot + new Vector3(0f, radius+Threshold, 0f), Threshold, Vector3.down, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.SphereCenterDown) {
				RaycastHit info;
				bool hit = Physics.SphereCast(pivot + new Vector3(0f, radius, 0f), Threshold, Vector3.down, out info, radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.SphereBottomUp) {
				RaycastHit info;
				bool hit = Physics.SphereCast(pivot - new Vector3(0f, radius+Threshold, 0f), Threshold, Vector3.up, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.SphereCenterUp) {
				RaycastHit info;
				bool hit = Physics.SphereCast(pivot - new Vector3(0f, radius, 0f), Threshold, Vector3.up, out info, radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayXPositive) {
				Vector3 dir = frame.GetBoneTransformation(Bone, mirrored).GetRight();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayXNegative) {
				Vector3 dir = -frame.GetBoneTransformation(Bone, mirrored).GetRight();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayYPositive) {
				Vector3 dir = frame.GetBoneTransformation(Bone, mirrored).GetUp();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayYNegative) {
				Vector3 dir = -frame.GetBoneTransformation(Bone, mirrored).GetUp();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayZPositive) {
				Vector3 dir = frame.GetBoneTransformation(Bone, mirrored).GetForward();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayZNegative) {
				Vector3 dir = -frame.GetBoneTransformation(Bone, mirrored).GetForward();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			collider = null;
			return pivot;
		}

		//TODO: FilterGaussian here has problems at the boundary of the file since the pivot point is not centered.
		public void CaptureContact(Frame frame, MotionEditor editor) {
			int width = Mathf.RoundToInt(Module.CaptureFilter * Module.Data.Framerate);
			Frame[] frames = Module.Data.GetFrames(Mathf.Clamp(frame.Index-width, 1, Module.Data.GetTotalFrames()), Mathf.Clamp(frame.Index+width, 1, Module.Data.GetTotalFrames()));
			{
				bool[] contacts = new bool[frames.Length];
				Vector3[] contactPoints = new Vector3[frames.Length];
				Vector3[] distances = new Vector3[frames.Length];
				for(int i=0; i<frames.Length; i++) {
					Frame f = frames[i];
					Vector3 bone = editor.Mirror ? f.GetBoneTransformation(Bone, false).GetPosition().GetMirror(f.Data.MirrorAxis) : f.GetBoneTransformation(Bone, false).GetPosition();
					Vector3 pivot = editor.Mirror ? GetPivot(f, false).GetMirror(f.Data.MirrorAxis) : GetPivot(f, false);
					Collider collider;
					Vector3 collision = DetectCollision(frame, false, Capture, pivot, Threshold, out collider);
					contacts[i] = collider != null;
					if(collider != null) {
						Vector3 distance = collision - bone;
						contactPoints[i] = editor.Mirror ? collision.GetMirror(f.Data.MirrorAxis) : collision;
						distances[i] = editor.Mirror ? distance.GetMirror(f.Data.MirrorAxis) : distance;
					}
				}
				bool hit = Utility.GetMostCommonItem(contacts);
				// Debug.Log(hit);
				if(hit) {
					RegularContacts[frame.Index-1] = 1f;
					RegularDistances[frame.Index-1] = Utility.GetMostCenteredVector(distances, contacts);
					RegularContactPoints[frame.Index-1] = Utility.GetMostCenteredVector(contactPoints, contacts);

				} else {
					RegularContacts[frame.Index-1] = 0f;
					RegularDistances[frame.Index-1] = Vector3.zero;
					RegularContactPoints[frame.Index-1] = Vector3.zero;
				}				
			}
			{
				bool[] contacts = new bool[frames.Length];
				Vector3[] distances = new Vector3[frames.Length];
				Vector3[] contactPoints = new Vector3[frames.Length];
				for(int i=0; i<frames.Length; i++) {
					Frame f = frames[i];
					Vector3 bone = editor.Mirror ? f.GetBoneTransformation(Bone, true).GetPosition() : f.GetBoneTransformation(Bone, true).GetPosition().GetMirror(f.Data.MirrorAxis);
					Vector3 pivot = editor.Mirror ? GetPivot(f, true) : GetPivot(f, true).GetMirror(f.Data.MirrorAxis);
					Collider collider;
					Vector3 collision = DetectCollision(frame, true, Capture, pivot, Threshold, out collider);
					contacts[i] = collider != null;
					if(collider != null) {
						Vector3 distance = collision - bone;
						distances[i] = editor.Mirror ? distance : distance.GetMirror(f.Data.MirrorAxis);
						contactPoints[i] = editor.Mirror ? collision : collision.GetMirror(f.Data.MirrorAxis);
					}
				}
				bool hit = Utility.GetMostCommonItem(contacts);
				if(hit) {
					InverseContacts[frame.Index-1] = 1f;
					InverseDistances[frame.Index-1] = Utility.GetMostCenteredVector(distances, contacts);
					InverseContactPoints[frame.Index-1] = Utility.GetMostCenteredVector(contactPoints, contacts);
				} else {
					InverseContacts[frame.Index-1] = 0f;
					InverseDistances[frame.Index-1] = Vector3.zero;
					InverseContactPoints[frame.Index-1] = Vector3.zero;
				}
			}
			if(Velocity > 0f) {
				if(GetContact(frame, false) == 1f) {
					if(frame.GetBoneVelocity(Bone, false, 1f/Module.Data.Framerate).magnitude > Velocity) {
						RegularContacts[frame.Index-1] = 0f;
						RegularContactPoints[frame.Index-1] = GetPivot(frame, false);
						RegularDistances[frame.Index-1] = Vector3.zero;
					}
				}
				if(GetContact(frame, true) == 1f) {
					if(frame.GetBoneVelocity(Bone, true, 1f/Module.Data.Framerate).magnitude > Velocity) {
						InverseContacts[frame.Index-1] = 0f;
						InverseContactPoints[frame.Index-1] = GetPivot(frame, true);
						InverseDistances[frame.Index-1] = Vector3.zero;
					}
				}
			}
			
		}

		public void ProcessContactPoints(bool mirrored) {
			CorrectedRegularGoalPoints = new Vector3[Module.Data.Frames.Length];
			CorrectedInverseGoalPoints = new Vector3[Module.Data.Frames.Length];

			ClusteredRegularGoalPoints = new Vector3[Module.Data.Frames.Length];
			ClusteredInverseGoalPoints = new Vector3[Module.Data.Frames.Length];

			reg_avg_goal_point = new List<Vector3>();
			inv_avg_goal_point = new List<Vector3>();

			List<Vector3> reg_point = new List<Vector3>();
			List<Vector3> inv_point = new List<Vector3>();

			List<int> reg_avg_index = new List<int>();
			List<int> inv_avg_index= new List<int>();
			
			List<Vector3> reg_count_point = new List<Vector3>();
			List<Vector3> inv_count_point = new List<Vector3>();

			bool avg_count_switch;
			int inv_original_count = 0;
			int reg_original_count = 0;

			bool is_ankle;
			if (this.GetName().Contains("Ankle")){
				is_ankle = true;
			}
			else{
				is_ankle = false;
			}

		
			// compute avergae contact point
			for(int i=1; i<Module.Data.Frames.Length-1; i++) {
				if (!is_ankle){
					if (mirrored){
						if (InverseContacts[i] == 1f & InverseContacts[i-1] == 0f & InverseContacts[i+1] == 1f) {
							inv_point.Add(InverseContactPoints[i]);
							inv_original_count++;
						}
					}
					else{
						if (RegularContacts[i-1] == 0f & RegularContacts[i] == 1f  & RegularContacts[i+1] == 1f) {
							reg_point.Add(RegularContactPoints[i]);
							// if (this.GetName() == "m_avg_R_Wrist"){
							// 	Debug.Log(i);
							// }
							reg_original_count ++;
						}
					}
				}
				if (i == reg_additional_key_frame){
					// Debug.Log(1);
					reg_point.Add(Module.Data.GetFrame(i).GetBoneTransformation(Bone, false).GetPosition());
				}
				if (i == inv_additional_key_frame){
					inv_point.Add(Module.Data.GetFrame(i).GetBoneTransformation(Bone, true).GetPosition());
				}
				// if (this.GetName() == "m_avg_L_Ankle"){
				// 	Debug.Log(string.Format("{0} ", reg_point.Count()));
				// } 
				// 	// Debug.Log(string.Format("{0} {1}", reg_avg_point.Count(), Module.Data.GetFrame(i).GetBoneTransformation(Bone, false).GetPosition()));
				// 	// Debug.Log(reg_avg_point[1]);

				
			}
			// if (this.GetName() == "m_avg_L_Ankle"){
			// 	Debug.Log(reg_point.Count());
			// 	Debug.Log(inv_point.Count());
			// }
			int reg_avg_contact_index;;
			int inv_avg_contact_index;
			// if (!is_ankle){
			reg_avg_contact_index = 0;
			inv_avg_contact_index = 0 ;
			// }
			// else{
			// 	reg_avg_contact_index = -1;
			// 	inv_avg_contact_index = -1 ;
			// }

			bool reg_contact_switch = false;
			bool inv_contact_switch = false;
			// Debug.Log(string.Format("Num of points {0}", reg_avg_point.Count));
			StyleModule StyleModule = ((StyleModule)Module.Data.Modules[2]);

			// Contact traj starts at the beginning of the style cycle, till contact finishes
			for(int i=10; i<Module.Data.Frames.Length; i++) {
				Frame frame = Module.Data.GetFrame(i);
				// Debug.Log(string.Format("{0} {1}", i, reg_avg_contact_index-1));

				if (mirrored){
				// Inverse
					if (i > 0){
						avg_count_switch = false;

						if ((StyleModule.GetStyle(frame, "Sit") > 0 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") == 0 & !is_ankle) |
							(Module.Sensors[0].InverseContacts[i] == 1 & Module.Sensors[0].InverseContacts[i-1] !=1  &  is_ankle)){
							if (inv_avg_contact_index  == 0) {
								if (inv_point.Count() > 0){
									// Debug.Log(1); 
									inv_avg_contact_index ++;
									inv_contact_switch = true;

									// Debug.Log(string.Format("{0} {1}", i, inv_avg_contact_index));			
									inv_count_point = new List<Vector3>();
									avg_count_switch = true;
								}
							}
						}

						if (InverseContacts[i] == 0f & InverseContacts[i-1] == 1f & InverseContacts[i-2] == 1f & !is_ankle){
							inv_contact_switch = false;
							avg_count_switch = true;

						}
						if (i== inv_additional_key_frame + 1){
							inv_contact_switch = false;
							avg_count_switch = true;

						}

						if (StyleModule.GetStyle(frame, "Sit") > 0 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") > 0 & !is_ankle){
							if (InverseContacts[i] == 0f & InverseContacts[i-1] == 1f & InverseContacts[i-2] == 1f | i == inv_additional_key_frame + 1){

								if (inv_avg_contact_index < inv_point.Count()){
									// if (this.GetName() == "m_avg_L_Wrist"){
									// 	Debug.Log(i);
									// }
									inv_contact_switch = true;
									inv_avg_contact_index ++;
								}
							}					
							// }
						}

						if (inv_contact_switch){
							if (InverseContacts[i] == 0f | is_ankle){
								CorrectedInverseGoalPoints[i] = inv_point[inv_avg_contact_index-1];
								inv_count_point.Add(CorrectedInverseGoalPoints[i]);

							}
							if (InverseContacts[i] == 1f & !is_ankle){
								CorrectedInverseGoalPoints[i] = GetCorrectedContactPoint(Module.Data.GetFrame(i), true);
								inv_count_point.Add(CorrectedInverseGoalPoints[i]);

							}
							

						}
						else{
							CorrectedInverseGoalPoints[i] = Vector3.zero;

						}
						if (avg_count_switch) {
							if (inv_count_point.Count>1){
								inv_avg_goal_point.Add(new Vector3(
									inv_count_point.Average(x=>x.x),
									inv_count_point.Average(x=>x.y),
									inv_count_point.Average(x=>x.z)));
								inv_count_point = new List<Vector3>();

							}
						}		
					}	
					else{
						CorrectedInverseGoalPoints[i] = Vector3.zero;
					}	
				}

				else{
					if (i > 0){
						avg_count_switch = false;
						if ((StyleModule.GetStyle(frame, "Sit") > 0 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") == 0 & !is_ankle) |
							(Module.Sensors[0].RegularContacts[i] == 1 & Module.Sensors[0].RegularContacts[i-1] != 1   &  is_ankle)){
							if (reg_avg_contact_index  == 0) {
								if (reg_point.Count() > 0){
									// if (this.GetName() == "m_avg_L_Wrist"){
									// 	Debug.Log(i);
									// }
									reg_avg_contact_index ++;
									reg_contact_switch = true;
									
									reg_count_point = new List<Vector3>();
									avg_count_switch = true;

								}
							}
						}

						if (RegularContacts[i] == 0f & RegularContacts[i-1] == 1f & RegularContacts[i-2] == 1f & !is_ankle){
							reg_contact_switch = false;
							avg_count_switch = true;

							
						}
						if (i== reg_additional_key_frame + 1){
							reg_contact_switch = false;
							avg_count_switch = true;

							// if (this.GetName() == "m_avg_L_Wrist"){
							// 		Debug.Log(reg_point.Count());
							// 	}

							
						}

						if (StyleModule.GetStyle(frame, "Sit") > 0 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") > 0 & !is_ankle){
							if (RegularContacts[i] == 0f & RegularContacts[i-1] == 1f & RegularContacts[i-2] == 1f | i == reg_additional_key_frame + 1){
								if (reg_avg_contact_index < reg_point.Count()){
									// if (this.GetName() == "m_avg_L_Wrist"){
									// 	Debug.Log(i);
									// }
									reg_contact_switch = true;
									reg_avg_contact_index ++;
									// avg_count_switch = true;
								 	// count_point = new List<Vector3>();

								}
							}					
						}
						if (reg_contact_switch){
							if (RegularContacts[i] == 0f | is_ankle){

								// if (this.GetName() == "m_avg_L_Wrist"){
								// 	Debug.Log(string.Format("{0} {1}/{2}", i, reg_avg_contact_index - 1, reg_point.Count()));
								// } 
								// if (this.GetName() == "m_avg_L_Wrist"){
								// Debug.Log(count_point.Count());
								// }
								CorrectedRegularGoalPoints[i] = reg_point[reg_avg_contact_index-1];
								reg_count_point.Add(CorrectedRegularGoalPoints[i]);

							}
							if (RegularContacts[i] == 1f & !is_ankle){
								// 	if (this.GetName() == "m_avg_R_Wrist"){
								// 	Debug.Log(count_point.Count());
								// }
								CorrectedRegularGoalPoints[i] = GetCorrectedContactPoint(Module.Data.GetFrame(i), false);
								reg_count_point.Add(CorrectedRegularGoalPoints[i]);

							}
							

						}
						else{


							CorrectedRegularGoalPoints[i] = Vector3.zero;

						}
						if (avg_count_switch) {
							if (reg_count_point.Count>1){
								reg_avg_goal_point.Add(new Vector3(
									reg_count_point.Average(x=>x.x),
									reg_count_point.Average(x=>x.y),
									reg_count_point.Average(x=>x.z)));
								// if (this.GetName() == "m_avg_L_Wrist"){
								// 	Debug.Log(string.Format("{0} {1}", i, count_point.Count));
								// }
								reg_count_point = new List<Vector3>();

							}

						}
								
					}	

					else{
						CorrectedRegularGoalPoints[i] = Vector3.zero;
					}	
				}
					
			

			}
		
		}
		// public void ProcessContactPoints(bool mirrored) {
		// 	CorrectedRegularGoalPoints = new Vector3[Module.Data.Frames.Length];
		// 	CorrectedInverseGoalPoints = new Vector3[Module.Data.Frames.Length];

		// 	ClusteredRegularGoalPoints = new Vector3[Module.Data.Frames.Length];
		// 	ClusteredInverseGoalPoints = new Vector3[Module.Data.Frames.Length];

		// 	reg_avg_goal_point = new List<Vector3>();
		// 	List<Vector3> reg_point = new List<Vector3>();
		// 	List<Vector3> inv_point = new List<Vector3>();

		// 	List<int> reg_avg_index = new List<int>();
		// 	List<int> inv_avg_index= new List<int>();
			
		// 	List<Vector3> count_point = new List<Vector3>();
		// 	bool avg_count_switch;
		// 	int inv_original_count = 0;
		// 	int reg_original_count = 0;

		// 	bool is_ankle;
		// 	if (this.GetName().Contains("Ankle")){
		// 		is_ankle = true;
		// 	}
		// 	else{
		// 		is_ankle = false;
		// 	}

		// 	// compute avergae contact point
		// 	for(int i=1; i<Module.Data.Frames.Length-1; i++) {
		// 		if (!is_ankle){
		// 			if (mirrored){
		// 				if (InverseContacts[i] == 1f & InverseContacts[i-1] == 0f & InverseContacts[i+1] == 1f) {
		// 					inv_point.Add(InverseContactPoints[i]);
		// 					inv_original_count++;
		// 				}
		// 			}
		// 			else{
		// 				if (RegularContacts[i-1] == 0f & RegularContacts[i] == 1f  & RegularContacts[i+1] == 1f) {
		// 					reg_point.Add(RegularContactPoints[i]);
		// 					// if (this.GetName() == "m_avg_R_Wrist"){
		// 					// 	Debug.Log(i);
		// 					// }
		// 					reg_original_count ++;
		// 				}
		// 			}
		// 		}
		// 		if (i == reg_additional_key_frame){
		// 			// Debug.Log(1);
		// 			reg_point.Add(Module.Data.GetFrame(i).GetBoneTransformation(Bone, false).GetPosition());
		// 		}
		// 		if (i == inv_additional_key_frame){
		// 			inv_point.Add(Module.Data.GetFrame(i).GetBoneTransformation(Bone, true).GetPosition());
		// 		}
		// 		// if (this.GetName() == "m_avg_L_Ankle"){
		// 		// 	Debug.Log(string.Format("{0} ", reg_point.Count()));
		// 		// } 
		// 		// 	// Debug.Log(string.Format("{0} {1}", reg_avg_point.Count(), Module.Data.GetFrame(i).GetBoneTransformation(Bone, false).GetPosition()));
		// 		// 	// Debug.Log(reg_avg_point[1]);

				
		// 	}
		// 	// if (this.GetName() == "m_avg_L_Ankle"){
		// 	// 	Debug.Log(reg_point.Count());
		// 	// 	Debug.Log(inv_point.Count());
		// 	// }
		// 	int reg_avg_contact_index;;
		// 	int inv_avg_contact_index;
		// 	// if (!is_ankle){
		// 	reg_avg_contact_index = 0;
		// 	inv_avg_contact_index = 0 ;
		// 	// }
		// 	// else{
		// 	// 	reg_avg_contact_index = -1;
		// 	// 	inv_avg_contact_index = -1 ;
		// 	// }

		// 	bool reg_contact_switch = false;
		// 	bool inv_contact_switch = false;
		// 	// Debug.Log(string.Format("Num of points {0}", reg_avg_point.Count));
		// 	StyleModule StyleModule = ((StyleModule)Module.Data.Modules[2]);

		// 	// Contact traj starts at the beginning of the style cycle, till contact finishes
		// 	for(int i=10; i<Module.Data.Frames.Length; i++) {
		// 		Frame frame = Module.Data.GetFrame(i);
		// 		// Debug.Log(string.Format("{0} {1}", i, reg_avg_contact_index-1));

		// 		if (mirrored){
		// 		// Inverse
		// 			if (i > 0){

		// 				if ((StyleModule.GetStyle(frame, "Sit") > 0 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") == 0 & !is_ankle) |
		// 					(StyleModule.GetStyle(frame, "Sit") ==1 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") < 1  &  is_ankle)){
		// 					if (inv_avg_contact_index < inv_point.Count()) {
		// 						// Debug.Log(1); 
		// 						inv_avg_contact_index ++;
		// 						// Debug.Log(string.Format("{0} {1}", i, inv_avg_contact_index));			
		// 						if (inv_point.Count() > 0){
		// 							inv_contact_switch = true;
		// 						}
		// 					}
		// 				}

		// 				if (InverseContacts[i] == 0f & InverseContacts[i-1] == 1f & InverseContacts[i-2] == 1f & !is_ankle){
		// 					inv_contact_switch = false;
		// 				}
		// 				if (i== inv_additional_key_frame + 1){
		// 					inv_contact_switch = false;
		// 				}

		// 				if (StyleModule.GetStyle(frame, "Sit") > 0 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") > 0 & !is_ankle){
		// 					if (InverseContacts[i] == 0f & InverseContacts[i-1] == 1f & InverseContacts[i-2] == 1f | i == inv_additional_key_frame + 1){

		// 						if (inv_avg_contact_index < inv_point.Count()){
		// 							// if (this.GetName() == "m_avg_L_Wrist"){
		// 							// 	Debug.Log(i);
		// 							// }
		// 							inv_contact_switch = true;
		// 							inv_avg_contact_index ++;
		// 						}
		// 					}					
		// 					// }
		// 				}

		// 				if (inv_contact_switch){
		// 					if (InverseContacts[i] == 0f){
		// 						CorrectedInverseGoalPoints[i] = inv_point[inv_avg_contact_index-1];
		// 					}
		// 					else{
		// 						CorrectedInverseGoalPoints[i] = GetCorrectedContactPoint(Module.Data.GetFrame(i), true);
		// 					}
		// 				}
		// 				else{
		// 					CorrectedInverseGoalPoints[i] = Vector3.zero;
		// 				}	
		// 			}	

		// 			else{
		// 				CorrectedInverseGoalPoints[i] = Vector3.zero;
		// 			}		
		// 		}

		// 		else{
		// 			if (i > 0){
		// 				avg_count_switch = false;
		// 				if ((StyleModule.GetStyle(frame, "Sit") > 0 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") == 0 & !is_ankle) |
		// 					(StyleModule.GetStyle(frame, "Sit") ==1 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") < 1  &  is_ankle)){
		// 					if (reg_avg_contact_index  == 0) {
		// 						if (reg_point.Count() > 0){
		// 							// if (this.GetName() == "m_avg_L_Wrist"){
		// 							// 	Debug.Log(i);
		// 							// }
		// 							reg_avg_contact_index ++;
		// 							reg_contact_switch = true;
									
		// 							count_point = new List<Vector3>();
		// 							avg_count_switch = true;

		// 						}
		// 					}
		// 				}

		// 				if (RegularContacts[i] == 0f & RegularContacts[i-1] == 1f & RegularContacts[i-2] == 1f & !is_ankle){
		// 					reg_contact_switch = false;
		// 					avg_count_switch = true;

							
		// 				}
		// 				if (i== reg_additional_key_frame + 1){
		// 					reg_contact_switch = false;
		// 					avg_count_switch = true;

		// 					// if (this.GetName() == "m_avg_L_Wrist"){
		// 					// 		Debug.Log(reg_point.Count());
		// 					// 	}

							
		// 				}

		// 				if (StyleModule.GetStyle(frame, "Sit") > 0 & StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") > 0 & !is_ankle){
		// 					if (RegularContacts[i] == 0f & RegularContacts[i-1] == 1f & RegularContacts[i-2] == 1f | i == reg_additional_key_frame + 1){
		// 						if (reg_avg_contact_index < reg_point.Count()){
		// 							// if (this.GetName() == "m_avg_L_Wrist"){
		// 							// 	Debug.Log(i);
		// 							// }
		// 							reg_contact_switch = true;
		// 							reg_avg_contact_index ++;
		// 							// avg_count_switch = true;
		// 						 	// count_point = new List<Vector3>();

		// 						}
		// 					}					
		// 				}
		// 				if (reg_contact_switch){
		// 					if (RegularContacts[i] == 0f | is_ankle){

		// 						// if (this.GetName() == "m_avg_L_Wrist"){
		// 						// 	Debug.Log(string.Format("{0} {1}/{2}", i, reg_avg_contact_index - 1, reg_point.Count()));
		// 						// } 
		// 						// if (this.GetName() == "m_avg_L_Wrist"){
		// 						// Debug.Log(count_point.Count());
		// 						// }
		// 						CorrectedRegularGoalPoints[i] = reg_point[reg_avg_contact_index-1];
		// 						count_point.Add(CorrectedRegularGoalPoints[i]);

		// 					}
		// 					if (RegularContacts[i] == 1f & !is_ankle){
		// 						// 	if (this.GetName() == "m_avg_R_Wrist"){
		// 						// 	Debug.Log(count_point.Count());
		// 						// }
		// 						CorrectedRegularGoalPoints[i] = GetCorrectedContactPoint(Module.Data.GetFrame(i), false);
		// 						count_point.Add(CorrectedRegularGoalPoints[i]);

		// 					}
							

		// 				}
		// 				else{


		// 					CorrectedRegularGoalPoints[i] = Vector3.zero;

		// 				}
		// 				if (avg_count_switch) {
		// 					if (count_point.Count>1){
		// 						reg_avg_goal_point.Add(new Vector3(
		// 							count_point.Average(x=>x.x),
		// 							count_point.Average(x=>x.y),
		// 							count_point.Average(x=>x.z)));
		// 						// if (this.GetName() == "m_avg_L_Wrist"){
		// 						// 	Debug.Log(string.Format("{0} {1}", i, count_point.Count));
		// 						// }
		// 						count_point = new List<Vector3>();

		// 					}

		// 				}
								
		// 			}	

		// 			else{
		// 				CorrectedRegularGoalPoints[i] = Vector3.zero;
		// 			}	
		// 		}
					
			

		// 	}
		
		// }

		public void ClusterContactPoints() {
			// Frame[] frames = Data.GetFrames();
			int count = 0;
			reg_avg_contact_point= new List<Vector3>();
			inv_avg_contact_point = new List<Vector3>();

			List<int> reg_avg_index = new List<int>();
			List<int> inv_avg_index= new List<int>();

			List<Vector3> reg_point = new List<Vector3>();
			List<Vector3> inv_point = new List<Vector3>();

			// compute avergae contact point
			for(int i=0; i<Module.Data.Frames.Length; i++) {
				if (RegularContacts[i] == 1f) {
					reg_point.Add(RegularContactPoints[i]);
					// reg_avg_index.Add(i);
				}
				else{
					if (reg_point.Count > 0){
						if (reg_point.Count > 1){
							reg_avg_contact_point.Add(new Vector3(
									reg_point.Average(x=>x.x),
									reg_point.Average(x=>x.y),
									reg_point.Average(x=>x.z)));
						}
						else {
							reg_avg_contact_point.Add(reg_point[0]);
							// Debug.Log(reg_point[0]);
						}
						
						reg_point = new List<Vector3>();
					}

				}
				if (InverseContacts[i] == 1f) {
					inv_point.Add(InverseContactPoints[i]);
				}
				else{
					if (inv_point.Count > 0){
						if (inv_point.Count > 1){
							inv_avg_contact_point.Add(new Vector3(
									inv_point.Average(x=>x.x),
									inv_point.Average(x=>x.y),
									inv_point.Average(x=>x.z)));
						}
						else {
							inv_avg_contact_point.Add(inv_point[0]);
						}
						inv_point = new List<Vector3>();
					}

				}

			}

			int reg_avg_contact_index = 0;
			int inv_avg_contact_index = 0 ;

			bool reg_contact_switch = false;
			bool inv_contact_switch = false;

			StyleModule StyleModule = ((StyleModule)Module.Data.Modules[2]);

			// Contact traj starts at the beginning of the style cycle, till contact finishes
			for(int i=0; i<Module.Data.Frames.Length; i++) {
				Frame frame = Module.Data.GetFrame(i);
				// Debug.Log(string.Format("{0} {1}", i, reg_avg_contact_index-1));

				if (i > 0){

					if (StyleModule.GetStyle(frame, "Sit") > 0){
						if (StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") == 0){
							// Debug.Log(1); 
							reg_avg_contact_index ++;
							// Debug.Log(string.Format("{0} {1}", i, reg_avg_contact_index));			

							if (reg_avg_contact_point.Count() > 0){
								reg_contact_switch = true;
							}
						}
					}

					if (RegularContacts[i] == 0f & RegularContacts[i-1] == 1f){
						// Debug.Log(2);
						reg_contact_switch = false;
					}
					if (StyleModule.GetStyle(frame, "Sit") == 1){
						if (StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") == 1){
							if (RegularContacts[i] == 0f & RegularContacts[i-1] == 1f){
								if (reg_avg_contact_point.Count() > reg_avg_contact_index){
									reg_contact_switch = true;
									reg_avg_contact_index ++;

								}					
							}
						}
					}

					if (reg_contact_switch){
						ClusteredRegularContactPoints[i] = reg_avg_contact_point[reg_avg_contact_index-1];
					}
					else{
						ClusteredRegularContactPoints[i] = Vector3.zero;
					}	
				}	

				else{
					ClusteredRegularContactPoints[i] = Vector3.zero;
				}	

					
			

				// Inverse
				if (i > 0){

					if (StyleModule.GetStyle(frame, "Sit") > 0){
						if (StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") == 0){
							// Debug.Log(1); 
							inv_avg_contact_index ++;
							// Debug.Log(string.Format("{0} {1}", i, inv_avg_contact_index));			

							if (inv_avg_contact_point.Count() > 0){
								inv_contact_switch = true;
							}
						}
					}

					if (InverseContacts[i] == 0f & InverseContacts[i-1] == 1f){
						// Debug.Log(2);
						inv_contact_switch = false;
					}
					if (StyleModule.GetStyle(frame, "Sit") == 1){
						if (StyleModule.GetStyle(Module.Data.GetFrame(i-1), "Sit") == 1){
							if (InverseContacts[i] == 0f & InverseContacts[i-1] == 1f){
								if (inv_avg_contact_point.Count() > inv_avg_contact_index){
									inv_contact_switch = true;
									inv_avg_contact_index ++;

								}					
							}
						}
					}

					if (inv_contact_switch){
						ClusteredInverseContactPoints[i] = inv_avg_contact_point[inv_avg_contact_index-1];
					}
					else{
						ClusteredInverseContactPoints[i] = Vector3.zero;
					}	
				}	

				else{
					ClusteredInverseContactPoints[i] = Vector3.zero;
				}	

						
			}
			
		}
		
		public void SetLocalPhaseByContacts(){
			
			List<int> RegularFramesToInterpolate = new List<int>();
			List<int> InverseFramesToInterpolate = new List<int>();
			

			StyleModule StyleModule = ((StyleModule)Module.Data.Modules[2]);

			bool is_ankle;
			if (this.GetName().Contains("Ankle")){
				is_ankle = true;
			}
			else{
				is_ankle = false;
			}
			
			if (!is_ankle){
				// Set Key Frames
				for(int i=2; i<Module.Data.Frames.Length; i++) {
					Frame frame = Module.Data.GetFrame(i);

					// start of sitting
					if (StyleModule.GetStyle(frame, "Sit") > 0 & RegularFramesToInterpolate.Count() == 0){
						// Debug.Log(i);
	// 
						RegularFramesToInterpolate.Add(i);
						InverseFramesToInterpolate.Add(i);
					}
					
					// every contact starts
					if (InverseContacts[i] == 1f & InverseContacts[i-1] == 0f){
						InverseFramesToInterpolate.Add(i);
					}

					// every contact finishes
					if (InverseContacts[i] == 0f & InverseContacts[i-1] == 1f){
						InverseFramesToInterpolate.Add(i);
					}

					// every contact starts
					if (RegularContacts[i] == 1f & RegularContacts[i-1] == 0f){
						// Debug.Log(i);
						RegularFramesToInterpolate.Add(i);
					}

					// every contact finishes
					if (RegularContacts[i] == 0f & RegularContacts[i-1] == 1f){
						// Debug.Log(i);
						RegularFramesToInterpolate.Add(i);
					}

					if (i == reg_additional_key_frame){
						// Debug.Log(i);
						RegularFramesToInterpolate.Add(i);
					}

					// every contact finishes
					if (i == inv_additional_key_frame){
						// Debug.Log(i);
						InverseFramesToInterpolate.Add(i);
					}



				}
			}
			else{
				if (reg_additional_key_frame!=0){
					for(int i=2; i<Module.Data.Frames.Length; i++) {
						Frame frame = Module.Data.GetFrame(i);
						if (Module.Sensors[0].InverseContacts[i] == 1 & Module.Sensors[0].InverseContacts[i-1] !=1 ){
							RegularFramesToInterpolate.Add(i);
							RegularFramesToInterpolate.Add(reg_additional_key_frame);
						}
					}
				}		
				if (inv_additional_key_frame!=0){
					for(int i=2; i<Module.Data.Frames.Length; i++) {
						Frame frame = Module.Data.GetFrame(i);
						if (Module.Sensors[0].InverseContacts[i] == 1 & Module.Sensors[0].InverseContacts[i-1] !=1 ){
							InverseFramesToInterpolate.Add(i);
							InverseFramesToInterpolate.Add(inv_additional_key_frame);
						}
					}
				}	
			}


			int InvCount = InverseFramesToInterpolate.Count();
			int RegCount = RegularFramesToInterpolate.Count();
			
			// fill beginning and end frames 

			// int regular_remainder_begin = 60 - RegularFramesToInterpolate[0] % 60;
			// int regular_remainder_end = 60 - (Module.Data.Frames.Length - RegularFramesToInterpolate[RegCount-1]) % 60;

			// int inverse_remainder_begin = 60 - InverseFramesToInterpolate[0] % 60;
			// int inverse_remainder_end = 60 - (Module.Data.Frames.Length - InverseFramesToInterpolate[InvCount-1]) % 60;

			// for(int i=0; i<Module.Data.Frames.Length; i++) {
				
			// 	// if (i < InverseFramesToInterpolate[0]){
			// 	// 	InverseLocalPhase[i] = (float) (i + inverse_remainder_begin) / 60f;
			// 	// }
			// 	if (i > InverseFramesToInterpolate[InvCount-1] ){
			// 		// Debug.Log( (float) (i - InverseFramesToInterpolate[InvCount-1] + inverse_remainder_end) % 60f / 60f);

			// 		InverseLocalPhase[i] = (float) (i - InverseFramesToInterpolate[InvCount-1] + inverse_remainder_end) % 60f / 60f;
			// 	}

			// 	// if (i < RegularFramesToInterpolate[0]){
			// 	// 	RegularLocalPhase[i] = (i + regular_remainder_begin) / 60f;
			// 	// }
			// 	if (i > RegularFramesToInterpolate[RegCount-1]){
			// 		RegularLocalPhase[i] = (float) (i - RegularFramesToInterpolate[RegCount-1] + regular_remainder_end) % 60f / 60f;
			// 	}
			// }

			// Debug.Log(RegularLocalPhase.Length);
			if (InvCount>1){
				// interpolation between key contact frames
				for (int j=0; j<InvCount - 1; j++){

					Interpolate(InverseLocalPhase, InverseFramesToInterpolate[j],  InverseFramesToInterpolate[j+1]);
				}
			}
			if (RegCount>1){
				

				for (int j=0; j<RegCount - 1; j++){

					Interpolate(RegularLocalPhase, RegularFramesToInterpolate[j],  RegularFramesToInterpolate[j+1]);
				}
			}
		}

		public void Interpolate(float[] values, int start, int end){

			for (int j=start; j< end; j++){
				values[j] =(float) (j - start) / (float)(end - start);
			}
		}

		public void Inspector(MotionEditor editor) {
			UltiDraw.Begin();
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Bone", GUILayout.Width(40f));
				Bone = EditorGUILayout.Popup(Bone, editor.GetData().Source.GetBoneNames(), GUILayout.Width(80f));
				EditorGUILayout.LabelField("Mask", GUILayout.Width(30));
				Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField(InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers, GUILayout.Width(75f)));
				EditorGUILayout.LabelField("Capture", GUILayout.Width(50));
				Capture = (ID)EditorGUILayout.EnumPopup(Capture, GUILayout.Width(75f));
				EditorGUILayout.LabelField("Edit", GUILayout.Width(30));
				Edit = (ID)EditorGUILayout.EnumPopup(Edit, GUILayout.Width(75f));
				EditorGUILayout.LabelField("Solve Position", GUILayout.Width(80f));
				SolvePosition = EditorGUILayout.Toggle(SolvePosition, GUILayout.Width(20f));
				EditorGUILayout.LabelField("Solve Rotation", GUILayout.Width(80f));
				SolveRotation = EditorGUILayout.Toggle(SolveRotation, GUILayout.Width(20f));
				EditorGUILayout.LabelField("Solve Distance", GUILayout.Width(80f));
				SolveDistance = EditorGUILayout.Toggle(SolveDistance, GUILayout.Width(20f));
				EditorGUILayout.EndHorizontal();

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Offset", GUILayout.Width(40f));
				Offset = EditorGUILayout.Vector3Field("", Offset, GUILayout.Width(180f));
				EditorGUILayout.LabelField("Threshold", GUILayout.Width(70f));
				Threshold = EditorGUILayout.FloatField(Threshold, GUILayout.Width(50f));
				EditorGUILayout.LabelField("Tolerance", GUILayout.Width(70f));
				Tolerance = EditorGUILayout.FloatField(Tolerance, GUILayout.Width(50f));
				EditorGUILayout.LabelField("Velocity", GUILayout.Width(70f));
				Velocity = EditorGUILayout.FloatField(Velocity, GUILayout.Width(50f));
				EditorGUILayout.LabelField("Weight", GUILayout.Width(60f));
				Weight = EditorGUILayout.FloatField(Weight, GUILayout.Width(50f));
				EditorGUILayout.EndHorizontal();

				Frame frame = editor.GetCurrentFrame();
				MotionData data = editor.GetData();

				EditorGUILayout.BeginVertical(GUILayout.Height(10f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 10f);
				EditorGUI.DrawRect(rect, UltiDraw.Black);

				float startTime = frame.Timestamp-editor.GetWindow()/2f;
				float endTime = frame.Timestamp+editor.GetWindow()/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > data.GetTotalTime()) {
					startTime -= endTime-data.GetTotalTime();
					endTime = data.GetTotalTime();
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(data.GetTotalTime(), endTime);
				int start = data.GetFrame(startTime).Index;
				int end = data.GetFrame(endTime).Index;
				int elements = end-start;

				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

				start = Mathf.Clamp(start, 1, Module.Data.Frames.Length);
				end = Mathf.Clamp(end, 1, Module.Data.Frames.Length);

				//Contacts
				for(int i=start; i<=end; i++) {
					if((editor.Mirror ? InverseContacts[i-1] : RegularContacts[i-1]) == 1f) {
						float left = rect.xMin + (float)(i-start)/(float)elements * rect.width;
						float right = left;
						while(i<end && (editor.Mirror ? InverseContacts[i-1] : RegularContacts[i-1]) != 0f) {
							right = rect.xMin + (float)(i-start)/(float)elements * rect.width;
							i++;
						}
						if(left != right) {
							Vector3 a = new Vector3(left, rect.y, 0f);
							Vector3 b = new Vector3(right, rect.y, 0f);
							Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
							Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
							UltiDraw.DrawTriangle(a, c, b, UltiDraw.Green);
							UltiDraw.DrawTriangle(b, c, d, UltiDraw.Green);
						}
					}
				}

				//Current Pivot
				top.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				top.y = rect.yMax - rect.height;
				bottom.y = rect.yMax;
				UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...

				EditorGUILayout.EndVertical();
			}
			UltiDraw.End();
		}
	}

}
#endif