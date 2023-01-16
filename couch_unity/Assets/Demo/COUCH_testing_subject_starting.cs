
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DeepLearning;
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using UnityEditorInternal;
using System.Linq;

public class  COUCH_testing_subject_starting: NeuralAnimation {
	public bool ShowSensorPositions = true;
	public bool ShowContactPositions = true;

	// private bool ShowIK = true;
	private Controller Controller;
	private TimeSeries TimeSeries;
	private TimeSeries.Root RootSeries;
	private TimeSeries.Style StyleSeries;
	private TimeSeries.Goal GoalSeries;
	private TimeSeries.Contact ContactSeries;
	private TimeSeries.Phase PhaseSeries;

	private CylinderMap Environment;
	private CuboidMap Geometry;

	private Frame frame;
	

	private Vector3[] PosePrediction;
	private Matrix4x4[] RootPrediction;
	private Matrix4x4[] GoalPrediction;
	private Vector3[][] SensorPositions;
	private Vector3[][] SensorPositions_x;

	private Vector3[] ContactPositions;

	private Vector3[] RetrievedPosition;
	private Vector3[] HandPosition;

	private int[] BoneMapping = new int[0];


	private float[] Signals = new float[0];
    private float UserControl = 0f;
	private float NetworkControl = 1f;

	private float InteractionSmoothing = 0.9f;

	private bool IsInteracting = false;

	private int StartFrameIndex;

	private UltimateIK.Model RootIK, RightFootIK, LeftFootIK, LeftHandIK, RightHandIK;

	// Initiating MotionData for sequences to be loaded	
	private MotionData[] Files = new MotionData[0];
	private MotionData File;

	private int frame_num;
	private int frame_num_init;

	private int frame_count = 0;


	private bool mirrored;
	private float delta;

	private int IK_iter = 5;
	private float IK_threshold = 0.1f;

	private Vector3[] SensorOffsets;

	private Vector3[] sensor_pos;

	private Vector3[] ik_targets;


	private int[] BoneIndexes;
	private List<string> BoneNames = new List<string> {"m_avg_R_Wrist", "m_avg_L_Wrist"};

	public ControlNetwork ControlNetwork;
	public List<Vector3> LeftContacts;
	public List<Vector3> RightContacts;

	private int[,] ContactIndex;
	private bool[,] InitialContactSwitch;



	public int TestingSequence = 17;

	private int chair_idx;
	private int rh_contact_idx;
	private int lh_contact_idx;
	public bool RightHandContact=true;
	public bool LeftHandContact=true;


	// testing switching from contact to no contact (lifting hands)
	private int NoContactCount = 0;
	private bool NoContactSwitch = false;

	private bool[] next_contacts;
	private Vector3[] ik_obj; 
	private Vector3[] ik_tar;

	private Transform target_root;


	protected override void ImportTestingSequence() {
		string[] assets = AssetDatabase.FindAssets("t:MotionData", new string[1]{"Assets/MotionCapture/testing"});
		File = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(assets[TestingSequence]), typeof(MotionData));
	}
	
	public Controller GetController() {
		return Controller;
	}

	public TimeSeries GetTimeSeries() {
		return TimeSeries;
	}

	public void UpdateTimeSeries(MotionData file, int frameNum, TimeSeries timeSeries, bool mir, float d){


		Frame frame = file.GetFrame(frameNum);
		foreach(Module module in file.Modules) {

			if(module is RootModule) {
				RootModule m = (RootModule)module;
				TimeSeries.Root series;
				series = new TimeSeries.Root(timeSeries);
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					if(t < 0f || t > file.GetTotalTime()) {
						series.Transformations[i] = m.GetEstimatedRootTransformation(frame, TimeSeries.Samples[i].Timestamp, mir);
						series.Velocities[i] = m.GetEstimatedRootVelocity(frame, TimeSeries.Samples[i].Timestamp, mir, d);
					} else {
						series.Transformations[i] = m.GetRootTransformation(file.GetFrame(t), mir);
						series.Velocities[i] = m.GetRootVelocity(file.GetFrame(t), mir, d);
					}
				}
			}
			if(module is StyleModule) {
				StyleModule m = (StyleModule)module;
				TimeSeries.Style series;
				series = new TimeSeries.Style(timeSeries, m.GetNames());
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					series.Values[i] = m.GetStyles(file.GetFrame(t));
				}
			}
			if(module is GoalModule) {
				GoalModule m = (GoalModule)module;
				TimeSeries.Goal series;
				series = new TimeSeries.Goal(timeSeries, m.GetNames());

				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;		
					series.Transformations[i] = m.Target.GetGoalTransformation(frame, timeSeries.Samples[i].Timestamp,  mir, d, false);
					series.Values[i] = m.GetActions(file.GetFrame(t), d);
					
				}
			}


			if(module is PhaseModule) {
				PhaseModule m = (PhaseModule)module;
				TimeSeries.Phase series;
				series = new TimeSeries.Phase(timeSeries);
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					series.Values[i] = m.GetPhase(file.GetFrame(t), mir);
				}
			}

			if(module is ContactModule) {
				ContactModule m = (ContactModule)module;
				
				TimeSeries.Contact series;
				series = new TimeSeries.Contact(timeSeries, m.GetNames());
				GoalModule m_goal = (GoalModule)file.GetModule(Module.ID.Goal);
				TimeSeries.Goal series_goal = new TimeSeries.Goal(timeSeries, m_goal.GetNames());


				for(int j=0; j<BoneNames.Count; j++){

					m.GetProcessedContacts(mirrored, BoneNames[j]);
					for(int i=0; i<timeSeries.Samples.Length; i++) {
						float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
						series.Values[i] = m.GetContacts(file.GetFrame(t), mirrored);

						series.Transformations[i][j] = Matrix4x4.TRS(m.GetCorrectedGoalPoint(file.GetFrame(t), mirrored, BoneNames[j]),  
																     m_goal.Target.GetGoalTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored, delta, false).GetRotation(),
																	 Vector3.one);
						series.CorrectedTransformations[i][j] = m.GetCurrentHandTarget(file.GetFrame(t), mirrored, BoneNames[j]);
						series.InitialCorrectedTransformations[i][j] = m.GetCurrentHandTarget(file.GetFrame(t), mirrored, BoneNames[j]);
						series.LocalPhases[i][j] = m.GetLocalPhases(file.GetFrame(t), mirrored, BoneNames[j]);

					}
				}
			}

		}
	}



	protected override void Setup() {
		Controller = new Controller();
		Controller.Signal idle = Controller.AddSignal("Idle");
		idle.Default = true;
		idle.Velocity = 0f;
		idle.AddKey(KeyCode.W, false);
		idle.AddKey(KeyCode.A, false);
		idle.AddKey(KeyCode.S, false);
		idle.AddKey(KeyCode.D, false);
		idle.AddKey(KeyCode.Q, false);
		idle.AddKey(KeyCode.E, false);
		idle.AddKey(KeyCode.V, true);
		// idle.UserControl = 0.25f;
		// idle.NetworkControl = 0.1f;
		idle.UserControl = 0f;
		idle.NetworkControl = 0.1f;

		Controller.Signal walk = Controller.AddSignal("Walk");
		walk.AddKey(KeyCode.W, true);
		walk.AddKey(KeyCode.A, true);
		walk.AddKey(KeyCode.S, true);
		walk.AddKey(KeyCode.D, true);
		walk.AddKey(KeyCode.Q, true);
		walk.AddKey(KeyCode.E, true);
		walk.AddKey(KeyCode.LeftShift, false);
		walk.AddKey(KeyCode.C, false);
		walk.Velocity = 1f;
		// walk.UserControl = 0.25f;
		// walk.NetworkControl = 0.25f;
		walk.UserControl = 0f;
		walk.NetworkControl = 0.25f;



		Controller.Signal sit = Controller.AddSignal("Sit");
		sit.AddKey(KeyCode.C, true);
		sit.Velocity = 0f;
		// sit.UserControl = 0.25f;
		// sit.NetworkControl = 0f;
		sit.UserControl = 0f;
		sit.NetworkControl = 0.9f;



		TimeSeries = new TimeSeries(6, 6, 1f, 1f, 5);
		float delta = 1f/30f;
		bool mirrored = false;
	
		((ContactModule)File.GetModule(Module.ID.Contact)).Sensors[1].SetAdditionalKeyFrame(0, 0);
		((ContactModule)File.GetModule(Module.ID.Contact)).Sensors[2].SetAdditionalKeyFrame(0, 0);
		((ContactModule)File.GetModule(Module.ID.Contact)).Sensors[1].SetLocalPhaseByContacts();
		((ContactModule)File.GetModule(Module.ID.Contact)).Sensors[2].SetLocalPhaseByContacts();


		StyleModule m =(StyleModule)File.GetModule(Module.ID.Style);
		// Getting the frame index where sitting style starts
		for (int i=0; i<File.Frames.Length; i++){
			if (m.GetStyle(File.GetFrame(i), "Idle") == 1f & m.GetStyle(File.GetFrame(i+1), "Idle") == 1f){
				frame_num ++ ;
			}
		} 
		frame_num = frame_num --;
		frame_num = frame_num;
		Frame frame = File.GetFrame(frame_num);

		SensorOffsets = ((ContactModule)File.GetModule(Module.ID.Contact)).GetSensorOffsets("m_avg_R_Wrist", "m_avg_L_Wrist", "m_avg_R_Ankle", "m_avg_L_Ankle");

		UpdateTimeSeries(File, frame_num, TimeSeries, mirrored, delta);
	
		RootSeries = (TimeSeries.Root)TimeSeries.GetSeries("Root");
		StyleSeries = (TimeSeries.Style)TimeSeries.GetSeries("Style");
		GoalSeries = (TimeSeries.Goal)TimeSeries.GetSeries("Goal");
		ContactSeries = (TimeSeries.Contact)TimeSeries.GetSeries("Contact");
		PhaseSeries = (TimeSeries.Phase)TimeSeries.GetSeries("Phase");
		Environment = ((CylinderMapModule)File.GetModule(Module.ID.CylinderMap)).GetCylinderMap(frame, mirrored);
		Geometry = ((GoalModule)File.GetModule(Module.ID.Goal)).Target.GetInteractionGeometry(frame, mirrored, delta);

		
		// Setting bone transformations from data in Frame to bones in Actor
		Actor.SetBoneTransformations(frame.GetBoneTransformations(false));
		BoneMapping = new int[Actor.Bones.Length];
		for(int i=0; i<Actor.Bones.Length; i++) {
			MotionData.Hierarchy.Bone bone = File.Source.FindBone(Actor.Bones[i].GetName());
			BoneMapping[i] = bone == null ? -1 : bone.Index;
		}
		// Scene scene = File.GetScene();
		// Frame frame = GetCurrentFrame();
		RootModule rootmodule = (RootModule)File.GetModule(Module.ID.Root);
		Matrix4x4 root = rootmodule == null ? frame.GetBoneTransformation(0, mirrored) : rootmodule.GetRootTransformation(frame, mirrored);
		Actor.transform.position = root.GetPosition();
		Actor.transform.rotation = root.GetRotation();
		// UpdateBoneMapping();s
		for(int i=0; i<Actor.Bones.Length; i++) {
			if(BoneMapping[i] == -1) {
				Debug.Log("Bone " + Actor.Bones[i].GetName() + " could not be mapped.");
			} else {
				Matrix4x4 transformation = frame.GetBoneTransformation(BoneMapping[i], mirrored);
				Vector3 velocity = frame.GetBoneVelocity(BoneMapping[i], mirrored, delta);
				Vector3 acceleration = frame.GetBoneAcceleration(BoneMapping[i],  mirrored, delta);
				Vector3 force = frame.GetBoneMass(BoneMapping[i], mirrored) * acceleration;
				Actor.Bones[i].Transform.position = transformation.GetPosition();
				Actor.Bones[i].Transform.rotation = transformation.GetRotation();
				Actor.Bones[i].Velocity = velocity;
				Actor.Bones[i].Acceleration = acceleration;
				Actor.Bones[i].Force = force;
			}
		}


		PosePrediction = new Vector3[Actor.Bones.Length];
		RootPrediction = new Matrix4x4[7];
		GoalPrediction = new Matrix4x4[7];
		HandPosition = new Vector3[BoneNames.Count];
		RetrievedPosition = new Vector3[BoneNames.Count];
		sensor_pos = new Vector3[BoneNames.Count];
		ik_targets = new Vector3[BoneNames.Count];


		RootIK = UltimateIK.BuildModel(Actor.FindTransform("m_avg_Pelvis"), Actor.GetBoneTransforms(ContactSeries.Bones[0]));
		RightHandIK = UltimateIK.BuildModel(Actor.FindTransform("m_avg_R_Shoulder"), Actor.GetBoneTransforms(ContactSeries.Bones[1]));
		LeftHandIK = UltimateIK.BuildModel(Actor.FindTransform("m_avg_L_Shoulder"), Actor.GetBoneTransforms(ContactSeries.Bones[2]));	
		RightFootIK = UltimateIK.BuildModel(Actor.FindTransform("m_avg_R_Hip"), Actor.GetBoneTransforms(ContactSeries.Bones[3]));
		LeftFootIK = UltimateIK.BuildModel(Actor.FindTransform("m_avg_L_Hip"), Actor.GetBoneTransforms(ContactSeries.Bones[4]));
		

		
		// Clustering contacts into descrete points and initialize for control

		ContactIndex = new int[TimeSeries.KeyCount, BoneNames.Count];
		InitialContactSwitch = new bool[TimeSeries.KeyCount, BoneNames.Count];
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			for(int j=0; j<BoneNames.Count; j++) {
				ContactIndex[i, j] = -1;
				InitialContactSwitch[i, j] = true;
			}
		}

	}

    protected override void Feed() {
		Controller.Update();

		//Get Root

		// TimeSeries.Pivot = 30; 
		Matrix4x4 root = RootSeries.Transformations[TimeSeries.Pivot];

		//Control Cycle
		Signals = Controller.PoolSignals();
		UserControl = Controller.PoolUserControl(Signals);
		NetworkControl = Controller.PoolNetworkControl(Signals);

		if(IsInteracting) {
			
			//Do nothing because coroutines have control.
		} else if(Controller.QuerySignal("Sit")) {
			StartCoroutine(Sit());
		} else {
			StartCoroutine(Sit());
			// Default();
		}

		//Input Bone Positions / Velocities
		for(int i=0; i<Actor.Bones.Length; i++) {
			PoseNetwork.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(root));
			PoseNetwork.Feed(Actor.Bones[i].Transform.forward.GetRelativeDirectionTo(root));
			PoseNetwork.Feed(Actor.Bones[i].Transform.up.GetRelativeDirectionTo(root));
			PoseNetwork.Feed(Actor.Bones[i].Velocity.GetRelativeDirectionTo(root));
		}

		//Input Trajectory Positions / Directions / Velocities / Styles
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			PoseNetwork.FeedXZ(RootSeries.GetPosition(sample.Index).GetRelativePositionTo(root));
			PoseNetwork.FeedXZ(RootSeries.GetDirection(sample.Index).GetRelativeDirectionTo(root));
			PoseNetwork.Feed(StyleSeries.Values[sample.Index]);
		}
		
		//Input Goals
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			PoseNetwork.Feed(GoalSeries.Transformations[sample.Index].GetPosition().GetRelativePositionTo(root));
			PoseNetwork.Feed(GoalSeries.Transformations[sample.Index].GetForward().GetRelativeDirectionTo(root));
			PoseNetwork.Feed(GoalSeries.Values[sample.Index]);
		}

		//Input Environment
		Environment.Sense(root, LayerMask.GetMask("Default", "Interaction"));
		PoseNetwork.Feed(Environment.Occupancies);

		//Input Geometry
		for(int i=0; i<Geometry.Points.Length; i++) {
			PoseNetwork.Feed(Geometry.References[i].GetRelativePositionTo(root));
			PoseNetwork.Feed(Geometry.Occupancies[i]);
		}

		
		SensorPositions = new Vector3[TimeSeries.KeyCount][];
		for(int k=0; k<TimeSeries.KeyCount; k++){
			SensorPositions[k] = new Vector3[BoneNames.Count];
		}

		SensorPositions_x = new Vector3[TimeSeries.KeyCount][];

		for(int k=0; k<TimeSeries.KeyCount; k++){
			SensorPositions_x[k] = new Vector3[BoneNames.Count];
		}

		ContactPositions = new Vector3[BoneNames.Count];

		// Input Hand Control
		for(int m=0; m<2; m++){
			// Input Hand Trajectory
			for(int k=0; k<BoneNames.Count; k++){
				Vector3 offset = SensorOffsets[k];

				if (ContactSeries.Transformations[TimeSeries.Pivot][k].GetPosition() != Vector3.zero){
					ContactPositions[k] = ContactSeries.Transformations[TimeSeries.Pivot][k].GetPosition();
					
				}
				for(int j=0; j<TimeSeries.KeyCount; j++) {
					TimeSeries.Sample sample = TimeSeries.GetKey(j);

					// Quaternion rot2 =  ContactSeries.GetCorrectedTransformation(TimeSeries.Pivot, k).GetRotation();
					// Vector3 SensorPosition = ContactSeries.GetCorrectedTransformation(sample.Index, k).GetPosition() + rot2 * offset;	
					// SensorPositions[j][k] = SensorPosition;

					if (ContactSeries.Transformations[sample.Index][k].GetPosition() == Vector3.zero){

						PoseNetwork.Feed(Vector3.zero);
					}
					else{
						Quaternion rot =  ContactSeries.GetCorrectedTransformation(TimeSeries.Pivot, k).GetRotation();
						Vector3 SensorPosition = ContactSeries.GetCorrectedTransformation(sample.Index, k).GetPosition() + rot * offset;	
						Vector3 tmp = SensorPosition.GetRelativePositionTo(ContactSeries.Transformations[sample.Index][k]);
						SensorPositions[j][k] = SensorPosition;

						
						if (NoContactSwitch == true){
							tmp = Vector3.zero;
						}


						PoseNetwork.Feed(tmp);

					}
				}
			}

			// Input Local Phase 
			for(int j=0; j<TimeSeries.KeyCount; j++) {
				TimeSeries.Sample sample = TimeSeries.GetKey(j);
				for (int k=0; k<BoneNames.Count; k++) {
					// float tmp = Mathf.Clamp(ContactSeries.Values[sample.Index][k+1], 0f, 1f);
					if (ContactSeries.Transformations[sample.Index][k].GetPosition() != Vector3.zero) {
						float tmp = ContactSeries.LocalPhases[sample.Index][k];
						PoseNetwork.Feed(tmp);
					}
					else{
						PoseNetwork.Feed(0f);
					}
					

				}
			}
		}

		//Setup Gating Features
		PoseNetwork.Feed(GenerateGating());
		

		/// ************** Feed Control Network *******************************
		ControlNetwork.ResetPivot();
		// Control Network Input: Feed Local Skeleton
		for(int i=0; i<Actor.Bones.Length; i++) {
			ControlNetwork.Feed(Actor.Bones[i].Transform.position.GetRelativePositionTo(root));
		}

		// Control Network Input: Trajectory
		for(int k=0; k<BoneNames.Count; k++){
			Vector3 offset = SensorOffsets[k];
			for(int j=TimeSeries.PivotKey; j<TimeSeries.KeyCount; j++) {
				TimeSeries.Sample sample = TimeSeries.GetKey(j);

				if (ContactSeries.Transformations[TimeSeries.Pivot][k].GetPosition() == Vector3.zero){
					ControlNetwork.Feed(Vector3.zero);
				}
				else{
					Quaternion rot =  ContactSeries.GetCorrectedTransformation(TimeSeries.Pivot, k).GetRotation();
					Vector3 SensorPosition = ContactSeries.GetCorrectedTransformation(TimeSeries.Pivot, k).GetPosition() + rot * offset;	
					float weight = (float)(TimeSeries.Samples.Length - j * 5 - 1) / TimeSeries.Pivot;
					Vector3 diff = (SensorPosition - ContactSeries.Transformations[TimeSeries.Pivot][k].GetPosition()) * weight;
					ControlNetwork.Feed(diff);
					SensorPositions_x[j][k] = diff + ContactSeries.Transformations[TimeSeries.Pivot][k].GetPosition();

				}
			}
		}


		
		// Control Network Input: Local Phase
		for(int b=0; b<BoneNames.Count; b++) {
			for(int k=TimeSeries.Pivot; k<TimeSeries.KeyCount; k++) {
				TimeSeries.Sample sample = TimeSeries.GetKey(k);
				// if (ContactSeries.Transformations[sample.Index][b].GetPosition() != Vector3.zero) {
					float tmp = ContactSeries.LocalPhases[sample.Index][b];
					PoseNetwork.Feed(tmp);
				// }
				// else{
				// 	PoseNetwork.Feed(0f);
				// }
			}
		}			
		
		ControlNetwork.Predict();
		ControlNetwork.ResetPivot();

	}

	protected override void Read() {
		frame_num ++;
		frame_count ++;
		if (frame_num == frame_num_init + 1){
			InitializeScene();
		}
		// UpdateTimeSeries(File, frame_num, TimeSeries, mirrored, delta, false);


		// Increment over States
		for(int i=0; i<TimeSeries.Pivot; i++) {
			// TimeSeries.Sample sample = TimeSeries.Samples[i];
			PhaseSeries.Values[i] = PhaseSeries.Values[i+1];
			RootSeries.SetPosition(i, RootSeries.GetPosition(i+1));
			RootSeries.SetDirection(i, RootSeries.GetDirection(i+1));
			for(int j=0; j<StyleSeries.Styles.Length; j++) {
				StyleSeries.Values[i][j] = StyleSeries.Values[i+1][j];
			}

			GoalSeries.Transformations[i] = GoalSeries.Transformations[i+1];
			for(int j=0; j<GoalSeries.Actions.Length; j++) {
				GoalSeries.Values[i][j] = GoalSeries.Values[i+1][j];
			}
			for(int j=0; j<BoneNames.Count; j++) { 
				ContactSeries.CorrectedTransformations[i][j] = ContactSeries.CorrectedTransformations[i+1][j];
				ContactSeries.Transformations[i][j] = ContactSeries.Transformations[i+1][j];
				ContactSeries.LocalPhases[i][j] = ContactSeries.LocalPhases[i+1][j];

			}			
			for(int j=0; j<ContactSeries.Bones.Length; j++) {
				ContactSeries.Values[i][j] = ContactSeries.Values[i+1][j];

			}
		
		}

		////////////////////////// Control Network ////////////////////////////////////
		Vector3 tmp;

		// Update Hand Trajectory
		for(int k=0; k<BoneNames.Count; k++){
			for(int j=TimeSeries.PivotKey; j<TimeSeries.KeyCount; j++) {
				TimeSeries.Sample sample = TimeSeries.GetKey(j);
				if (ContactSeries.Transformations[sample.Index][k].GetPosition() == Vector3.zero){
					tmp = ControlNetwork.ReadVector3();
					ContactSeries.SetCorrectedPosition(sample.Index, tmp.GetRelativePositionFrom(ContactSeries.Transformations[sample.Index][k]), k);

				}
				else{
					// visualise hands
					Quaternion rot =  ContactSeries.GetCorrectedTransformation(j, k).GetRotation();
					tmp = ControlNetwork.ReadVector3() - rot * SensorOffsets[k];

					// Set Hand Positions
					ContactSeries.SetCorrectedPosition(sample.Index, tmp.GetRelativePositionFrom(ContactSeries.Transformations[sample.Index][k]), k);
				}
				if (j == TimeSeries.PivotKey){
					ik_targets[k]  = tmp.GetRelativePositionFrom(ContactSeries.Transformations[sample.Index][k]);

				}
			}
		}


		// Update Local Phases
		for (int k=0; k<BoneNames.Count; k++) {
			float phase = ContactSeries.LocalPhases[TimeSeries.Pivot][k];
			for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
				ContactSeries.LocalPhases[TimeSeries.GetKey(i).Index][k] = Mathf.Repeat(phase + ControlNetwork.Read(), 1f);
			}
		}

	

		// // Interpolation
		// for (int j=0; j<BoneNames.Count; j++){
		// 	for(int i=TimeSeries.Pivot ; i<TimeSeries.Samples.Length; i++) {
		// 		float weight = (float)(i % TimeSeries.Resolution) / TimeSeries.Resolution;
		// 		TimeSeries.Sample sample = TimeSeries.Samples[i];
		// 		TimeSeries.Sample prevSample = TimeSeries.GetPreviousKey(i);
		// 		TimeSeries.Sample nextSample = TimeSeries.GetNextKey(i);
		// 		ContactSeries.SetCorrectedPosition(sample.Index, Vector3.Lerp(ContactSeries.CorrectedTransformations[prevSample.Index][j].GetPosition(), ContactSeries.CorrectedTransformations[nextSample.Index][j].GetPosition(), weight), j);
		// 		ContactSeries.Transformations[i][j] = Utility.Interpolate(ContactSeries.Transformations[prevSample.Index][j], ContactSeries.Transformations[nextSample.Index][j], weight);
		// 	}
		// }

		// Get Root
		Matrix4x4 root = RootSeries.Transformations[TimeSeries.Pivot];


		//Read Posture
		Vector3[] positions = new Vector3[Actor.Bones.Length];
		Vector3[] forwards = new Vector3[Actor.Bones.Length];
		Vector3[] upwards = new Vector3[Actor.Bones.Length];
		Vector3[] velocities = new Vector3[Actor.Bones.Length]; 
		sensor_pos = new Vector3[BoneNames.Count];

		for(int i=0; i<Actor.Bones.Length; i++) {
			Vector3 position = PoseNetwork.ReadVector3().GetRelativePositionFrom(root);
			Vector3 forward = PoseNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 upward = PoseNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			Vector3 velocity = PoseNetwork.ReadVector3().GetRelativeDirectionFrom(root);
			positions[i] = Vector3.Lerp(Actor.Bones[i].Transform.position + velocity / GetFramerate(), position, 0.5f);
			forwards[i] = forward;
			upwards[i] = upward;
			velocities[i] = velocity;

			// Update Hand Trajectories
			for (int b=0; b<BoneNames.Count; b++) {
				if (Actor.Bones[i].GetName() == BoneNames[b]){
					ContactSeries.SetCorrectedPosition(TimeSeries.Pivot, positions[i], b);
					ContactSeries.SetCorrectedRotation(TimeSeries.Pivot, Quaternion.LookRotation(forwards[i], upwards[i]), b);
					sensor_pos[b] = positions[i] + Quaternion.LookRotation(forwards[i], upwards[i]) * SensorOffsets[b];
				}
			}
		}

		
		//Read Inverse Pose
		for(int i=0; i<Actor.Bones.Length; i++) {
			PosePrediction[i] = PoseNetwork.ReadVector3().GetRelativePositionFrom(RootSeries.Transformations.Last());
			velocities[i] = Vector3.Lerp(velocities[i], GetFramerate() * (PosePrediction[i] - Actor.Bones[i].Transform.position), 1f/GetFramerate());
		}

		// //Read Future Trajectory
		for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			Vector3 pos = PoseNetwork.ReadXZ().GetRelativePositionFrom(root);
			Vector3 dir = PoseNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(root);
			RootSeries.SetPosition(sample.Index, pos);
			RootSeries.SetDirection(sample.Index, dir);
			float[] styles = PoseNetwork.Read(StyleSeries.Styles.Length);
			
			for(int j=0; j<styles.Length; j++) {
				styles[j] = Mathf.Clamp(styles[j], 0f, 1f);
			}

			StyleSeries.Values[sample.Index] = styles;
		
			RootPrediction[i-6] = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
		}


		// //Read Inverse Trajectory
		for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			Matrix4x4 goal = GoalSeries.Transformations[TimeSeries.Pivot];
			goal[1,3] = 0f;
			Vector3 pos = PoseNetwork.ReadXZ().GetRelativePositionFrom(goal);
			Vector3 dir = PoseNetwork.ReadXZ().normalized.GetRelativeDirectionFrom(goal);
			if(i > TimeSeries.PivotKey) {
				Matrix4x4 pivot = RootSeries.Transformations[sample.Index];
				pivot[1,3] = 0f;
				Matrix4x4 reference = GoalSeries.Transformations[sample.Index];
				reference[1,3] = 0f;
				float distance = Vector3.Distance(pivot.GetPosition(), reference.GetPosition());
				float weight = Mathf.Pow((float)(i-6)/7f, distance*distance);

				RootSeries.SetPosition(sample.Index, Vector3.Lerp(RootSeries.GetPosition(sample.Index), pos, weight));
				RootSeries.SetDirection(sample.Index, Vector3.Slerp(RootSeries.GetDirection(sample.Index), dir, weight));
			}
			
			GoalPrediction[i-6] = Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one);
		}
		
		// //Read and Correct Goals
		for(int i=0; i<TimeSeries.KeyCount; i++) {
			float weight = TimeSeries.GetWeight1byN1(TimeSeries.GetKey(i).Index, 2f);
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			Vector3 pos = PoseNetwork.ReadVector3().GetRelativePositionFrom(root);
			Vector3 dir = PoseNetwork.ReadVector3().normalized.GetRelativeDirectionFrom(root);
			float[] actions = PoseNetwork.Read(GoalSeries.Actions.Length);
			for(int j=0; j<actions.Length; j++) {
				actions[j] = Mathf.Clamp(actions[j], 0f, 1f);
			}
			GoalSeries.Transformations[sample.Index] = Utility.Interpolate(GoalSeries.Transformations[sample.Index], Matrix4x4.TRS(pos, Quaternion.LookRotation(dir, Vector3.up), Vector3.one), weight * NetworkControl);
			GoalSeries.Values[sample.Index] = Utility.Interpolate(GoalSeries.Values[sample.Index], actions, weight * NetworkControl);
		}


		// //Read Future Contacts
		float[] contacts = PoseNetwork.Read(ContactSeries.Bones.Length);
		for(int i=0; i<contacts.Length; i++) {
			contacts[i] = Mathf.Clamp(contacts[i], 0f, 1f);
		}
		ContactSeries.Values[TimeSeries.Pivot] = contacts;

		//Read Phase Update
		float phase_global = PhaseSeries.Values[TimeSeries.Pivot];
		for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
			PhaseSeries.Values[TimeSeries.GetKey(i).Index] = Mathf.Repeat(phase_global + PoseNetwork.Read(), 1f);
		}

		// Interpolate Current to Future Trajectory
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			float weight = (float)(i % TimeSeries.Resolution) / TimeSeries.Resolution;
			TimeSeries.Sample sample = TimeSeries.Samples[i];
			TimeSeries.Sample prevSample = TimeSeries.GetPreviousKey(i);
			TimeSeries.Sample nextSample = TimeSeries.GetNextKey(i);
			//PhaseSeries.Values[sample.Index] = Mathf.Lerp(PhaseSeries.Values[prevSample.Index], PhaseSeries.Values[nextSample.Index], weight);
			RootSeries.SetPosition(sample.Index, Vector3.Lerp(RootSeries.GetPosition(prevSample.Index), RootSeries.GetPosition(nextSample.Index), weight));
			RootSeries.SetDirection(sample.Index, Vector3.Slerp(RootSeries.GetDirection(prevSample.Index), RootSeries.GetDirection(nextSample.Index), weight));
			GoalSeries.Transformations[sample.Index] = Utility.Interpolate(GoalSeries.Transformations[prevSample.Index], GoalSeries.Transformations[nextSample.Index], weight);
			for(int j=0; j<StyleSeries.Styles.Length; j++) {
				StyleSeries.Values[i][j] = Mathf.Lerp(StyleSeries.Values[prevSample.Index][j], StyleSeries.Values[nextSample.Index][j], weight);
			}
			for(int j=0; j<GoalSeries.Actions.Length; j++) {
				GoalSeries.Values[i][j] = Mathf.Lerp(GoalSeries.Values[prevSample.Index][j], GoalSeries.Values[nextSample.Index][j], weight);

			}
		}

		//Assign Posture
		transform.position = RootSeries.GetPosition(TimeSeries.Pivot);
		transform.rotation = RootSeries.GetRotation(TimeSeries.Pivot);
		for(int i=0; i<Actor.Bones.Length; i++) {
			Actor.Bones[i].Velocity = velocities[i];
			Actor.Bones[i].Transform.position = positions[i];
			Actor.Bones[i].Transform.rotation = Quaternion.LookRotation(forwards[i], upwards[i]);
			Actor.Bones[i].ApplyLength();
		}




		// Getting Current Contact Points from the Data (does not matter now if it's out of sync with GT)
		for (int j=0; j<BoneNames.Count; j++){
			UpdateContactIndex(TimeSeries.PivotKey, j);
		}
		for (int j=0; j<BoneNames.Count; j++){
			Vector3 contact = Vector3.zero;
			for(int i=TimeSeries.PivotKey; i<TimeSeries.KeyCount; i++) {
				TimeSeries.Sample sample = TimeSeries.GetKey(i);

					int idx = ContactIndex[i, j];
					if (idx != -1){
						if (j == 0){ 
							if (idx < RightContacts.Count()){
								contact = RightContacts[idx];

							}
							else{
								contact = RightContacts[RightContacts.Count() - 1];
							}
						}
						if (j == 1){
							if (idx < LeftContacts.Count()){

								contact = LeftContacts[idx];


							}
							else{
								contact = LeftContacts[LeftContacts.Count() - 1];
							}
						}
						
					}

				ContactSeries.Transformations[sample.Index][j] = Matrix4x4.TRS(contact, 
																			   GoalSeries.GetRotation(TimeSeries.Pivot),
																			   Vector3.one);
			}
		
		}
	}




	private void UpdateContactIndex(int TimeIndex, int BoneIndex){
		int threshold = ContactIndex[TimeSeries.PivotKey, BoneIndex] +1;
		for(int i=TimeIndex; i<TimeSeries.KeyCount; i++) {
			TimeSeries.Sample sample = TimeSeries.GetKey(i);
			TimeSeries.Sample prev_sample = TimeSeries.GetKey(i-1);

			if (StyleSeries.GetStyle(sample.Index, "Sit") > 0.1f & InitialContactSwitch[i, BoneIndex] == true){
				ContactIndex[i, BoneIndex] ++;
				InitialContactSwitch[i, BoneIndex] = false;
			}
			if ((ContactSeries.Transformations[TimeSeries.Pivot][BoneIndex].GetPosition() - sensor_pos[BoneIndex]).magnitude < 0.2f){

				// if (ContactSeries.LocalPhases[prev_sample.Index][BoneIndex] - ContactSeries.LocalPhases[sample.Index][BoneIndex] > 0.5f){
					// min_contact[BoneIndex] = 
					for (int j=i; j<TimeSeries.KeyCount; j++){
						// Debug.Log(string.Format("{0} {1}", ContactIndex[j, BoneIndex] - ContactIndex[TimeSeries.PivotKey, BoneIndex], ContactIndex[j, BoneIndex]));
						if (threshold - ContactIndex[j, BoneIndex] == 1){   
							ContactIndex[j, BoneIndex] ++;
							// if (BoneIndex==0){
								// Debug.Log(string.Format("{0} {1} {2} {3}", frame_num, BoneIndex, j, ContactIndex[j, BoneIndex]));
							// }
						}


					}
					break;
				// } 
			}
			// UpdateContactIndex(i+1, BoneIndex);

		}
	}
	

    protected override void Postprocess() {
		Actor.RestoreAlignment();

		
		if (ContactSeries.Values[TimeSeries.Pivot][0] > 0.98f){

			Matrix4x4 root = Actor.GetBoneTransformation(ContactSeries.Bones[0]);
			RootIK.Objectives[0].SetTarget(target_root.position, 1);
			// RootIK.Objectives[0].SetTarget(Vector3.Lerp(RootIK.Bones[RootIK.Objectives[0].Bone].Transform.position, target_lh, ContactSeries.Values[TimeSeries.Pivot][0]));
			// RootIK.Objectives[0].SetTarget(root.GetPosition(), 1f-ContactSeries.Values[TimeSeries.Pivot][0]);
			RootIK.Objectives[0].SetTarget(target_root.rotation);
			RootIK.Solve();
		}
		else{
			target_root = RootIK.Bones[RootIK.Objectives[0].Bone].Transform;

		}

		Matrix4x4 rightFoot = Actor.GetBoneTransformation(ContactSeries.Bones[3]);
		Matrix4x4 leftFoot = Actor.GetBoneTransformation(ContactSeries.Bones[4]);
		RightFootIK.Objectives[0].SetTarget(rightFoot.GetPosition(), 1f-ContactSeries.Values[TimeSeries.Pivot][3]);
		RightFootIK.Objectives[0].SetTarget(rightFoot.GetRotation());
		LeftFootIK.Objectives[0].SetTarget(leftFoot.GetPosition(), 1f-ContactSeries.Values[TimeSeries.Pivot][4]);
		LeftFootIK.Objectives[0].SetTarget(leftFoot.GetRotation());
		RightFootIK.Solve();
		LeftFootIK.Solve();

		// Transform rightToe = Actor.FindBone("RightToe").Transform;
		Transform rightToe = Actor.FindBone("m_avg_R_Foot").Transform;

		Vector3 rightPos = rightToe.transform.position;
		rightPos.y = Mathf.Max(rightPos.y, 0.07f);
		rightToe.position = rightPos;

		// Transform leftToe = Actor.FindBone("LeftToe").Transform;
		Transform leftToe = Actor.FindBone("m_avg_L_Foot").Transform;

		Vector3 leftPos = leftToe.transform.position;
		leftPos.y = Mathf.Max(leftPos.y, 0.07f);
		leftToe.position = leftPos;
		
		/////////// fitting hands
		ik_obj = new Vector3[2];
		ik_tar = new Vector3[2];

		if (ContactSeries.Transformations[TimeSeries.Pivot][0].GetPosition() != Vector3.zero & GoalSeries.GetAction(TimeSeries.Pivot, "Sit") > 0.9f){
			Vector3 rh_contact = ContactSeries.Transformations[TimeSeries.Pivot][0].GetPosition() -  ContactSeries.CorrectedTransformations[TimeSeries.Pivot][0].GetRotation() * SensorOffsets[0];

			RightHandIK.Activation = UltimateIK.ACTIVATION.Constant;
			float distance =(RightHandIK.Bones[RightHandIK.Objectives[0].Bone].Transform.position - rh_contact).magnitude;
			Vector3 target_rh;
			if (distance > 0.5){
				// target_rh =  ContactSeries.CorrectedTransformations[TimeSeries.Pivot][0].GetPosition();
				target_rh =  ik_targets[0];

			}
			else{
				target_rh = rh_contact;
			}
			// RightHandIK.Objectives[0].SetTarget(target_rh, 1);
			if (distance > 0.5){
				RightHandIK.Objectives[0].SetTarget(Vector3.Lerp(RightHandIK.Bones[RightHandIK.Objectives[0].Bone].Transform.position, target_rh, ContactSeries.Values[TimeSeries.Pivot][1]));
			}
			else{
				RightHandIK.Objectives[0].SetTarget(Vector3.Lerp(RightHandIK.Bones[RightHandIK.Objectives[0].Bone].Transform.position, target_rh, ContactSeries.Values[TimeSeries.Pivot][1]));

			}

			RightHandIK.Objectives[0].SetTarget(RightHandIK.Bones[RightHandIK.Objectives[0].Bone].Transform.rotation);


			RightHandIK.Iterations = 50;
			RightHandIK.Solve();
			ik_obj[0] =  RightHandIK.Bones[RightHandIK.Objectives[0].Bone].Transform.position;
			ik_tar[0] =  RightHandIK.Objectives[0].TargetPosition;
		}

		if (ContactSeries.Transformations[TimeSeries.Pivot][1].GetPosition() != Vector3.zero & GoalSeries.GetAction(TimeSeries.Pivot, "Sit") > 0.9f){

			Vector3 lh_contact = ContactSeries.Transformations[TimeSeries.Pivot][1].GetPosition() -  ContactSeries.CorrectedTransformations[TimeSeries.Pivot][1].GetRotation() * SensorOffsets[1];

			LeftHandIK.Activation = UltimateIK.ACTIVATION.Constant;
			float distance =(LeftHandIK.Bones[LeftHandIK.Objectives[0].Bone].Transform.position - lh_contact).magnitude;

			Vector3 target_lh;
			if (distance > 0.5){
				target_lh =  ik_targets[1];

			}
			else{
				target_lh = lh_contact;
			}
			// RightHandIK.Objectives[0].SetTarget(target_rh, 1);
			if (distance > 0.5){
				LeftHandIK.Objectives[0].SetTarget(Vector3.Lerp(LeftHandIK.Bones[LeftHandIK.Objectives[0].Bone].Transform.position, target_lh, ContactSeries.Values[TimeSeries.Pivot][2]));
				}
			else{				
				LeftHandIK.Objectives[0].SetTarget(Vector3.Lerp(LeftHandIK.Bones[LeftHandIK.Objectives[0].Bone].Transform.position, target_lh, ContactSeries.Values[TimeSeries.Pivot][2]));
			}


			// LeftHandIK.Objectives[0].SetTarget(Vector3.Lerp(LeftHandIK.Bones[LeftHandIK.Objectives[0].Bone].Transform.position, target_lh, ContactSeries.Values[TimeSeries.Pivot][2]));
			LeftHandIK.Objectives[0].SetTarget(LeftHandIK.Bones[LeftHandIK.Objectives[0].Bone].Transform.rotation);


			LeftHandIK.Iterations = 50;
			LeftHandIK.Solve();
			ik_obj[1] = LeftHandIK.Bones[LeftHandIK.Objectives[0].Bone].Transform.position;
			ik_tar[1] =  LeftHandIK.Objectives[0].TargetPosition;
		}
    }


	protected override IEnumerator InitializeScene(){
		while(!File.GetScene().isLoaded) {
			Debug.Log("Waiting for scene to be loaded.");
			yield return new WaitForSeconds(0f);
		}
		// Disable Original Scene
		GameObject[] gameObjects = Utility.Unroll(File.GetScene().GetRootGameObjects());
		gameObjects[0].SetActive(false);

		// Create Selected/Sampled Scene 
		string [] obj_files = System.IO.Directory.GetFiles("Assets/Resources/Assets/chairs", "*.prefab");
		string tmp = obj_files[0].Split('/').Last().Split('.')[0];
		List<string> chairs =  new List<string> {};
		GameObject instance = Instantiate(Resources.Load("Assets/chairs" + '/' + tmp, typeof(GameObject))) as GameObject;
		for (int i=0; i<instance.transform.childCount; i++){
			GameObject gameObject = instance.transform.GetChild(i).gameObject;
			if(gameObject.GetComponent<Interaction>() != null){
				chairs.Add(gameObject.name);	
			}
		}


		string date_chair = tmp;


		// Random Sampling Chair and Contact
		System.Random rnd = new System.Random();
		chair_idx  = rnd.Next(0, instance.transform.childCount-1);	
		
		string chair_name = chairs[chair_idx];
		string session_name = chair_name.Split('_')[0];
		string asset_name = chair_name.Split('_')[1];
		string which_chair = chair_name.Split('_')[2];

		string lh_contact_file = "Assets/MotionCapture/Contacts/lh" + '/' + session_name + '_' +  asset_name + '/' + which_chair + ".txt";
		string[] lh_contacts =  System.IO.File.ReadAllLines(lh_contact_file);
		string rh_contact_file = "Assets/MotionCapture/Contacts/rh" + '/' + session_name + '_' +  asset_name + '/' + which_chair + ".txt";
		string[] rh_contacts =  System.IO.File.ReadAllLines(rh_contact_file);

		lh_contact_idx = rnd.Next(0, lh_contacts.Length-1);
		rh_contact_idx = rnd.Next(0, rh_contacts.Length-1);
		
		if ((float)rnd.NextDouble() > 0.5f){
			LeftHandContact = true;
		}
		if ((float)rnd.NextDouble() > 0.5f){
			RightHandContact = true;
		}


		// LH Contact
		float[][] lh_contacts_all = lh_contacts.Select(line => Array.ConvertAll(line.Split(' ').ToArray(), float.Parse)).ToArray();
		Vector3 lh_contact = new Vector3(lh_contacts_all[lh_contact_idx][0], 
										 lh_contacts_all[lh_contact_idx][1],
										 lh_contacts_all[lh_contact_idx][2]);

		


		// RH Contact
		float[][] rh_contacts_all = rh_contacts.Select(line => Array.ConvertAll(line.Split(' ').ToArray(), float.Parse)).ToArray();
		Vector3 rh_contact = new Vector3(rh_contacts_all[rh_contact_idx][0], 
										 rh_contacts_all[rh_contact_idx][1],
										 rh_contacts_all[rh_contact_idx][2]);
		

		// Activate Chair and Initialize Contact Input
		for (int i=0; i<instance.transform.childCount; i++){
			GameObject gameObject = instance.transform.GetChild(i).gameObject;
			if(gameObject.GetComponent<Interaction>() != null){

				// if (gameObject.name.Split('_').Last() + ".txt" == rh_contact_file.Split('\\').Last()){
				if (chair_idx == i){

					rh_contact += gameObject.GetComponent<Interaction>().GetCenter().GetPosition();
					lh_contact += gameObject.GetComponent<Interaction>().GetCenter().GetPosition();

					gameObject.SetActive(true);
				}
				else{
					gameObject.SetActive(false);

				}
			}
		}
		
		if (LeftHandContact == true){
			LeftContacts.Add(lh_contact);
		}
		else{
			LeftContacts.Add(Vector3.zero);

		
		}
		if (RightHandContact == true){
			RightContacts.Add(rh_contact);
		}
		else{
			RightContacts.Add(Vector3.zero);
		}

		
	}
	private void Default() {
		if(Controller.ProjectionActive) {
			
			ApplyStaticGoal(Controller.Projection.point, Vector3.ProjectOnPlane(Controller.Projection.point-transform.position, Vector3.up).normalized, Signals);

		} else {
			// Debug.Log(2);
			ApplyDynamicGoal(
				RootSeries.Transformations[TimeSeries.Pivot],
				Controller.QueryMove(KeyCode.W, KeyCode.S, KeyCode.A, KeyCode.D, Signals),
				Controller.QueryTurn(KeyCode.Q, KeyCode.E, 90f), 
				Signals
			);
		}
		Geometry.Setup(Geometry.Resolution);
		Geometry.Sense(RootSeries.Transformations[TimeSeries.Pivot], LayerMask.GetMask("Interaction"), Vector3.zero, InteractionSmoothing);
	}

	private IEnumerator Sit() {

		Controller.Signal signal = Controller.GetSignal("Sit");
		Interaction interaction = Controller.ProjectionInteraction != null ? Controller.ProjectionInteraction : Controller.GetClosestInteraction(transform);
		float threshold = 0.25f;

		if(interaction != null) {
			
			Controller.ActiveInteraction = interaction;
			IsInteracting = true;
			ApplyStaticGoal(interaction.GetContact("m_avg_Pelvis").GetPosition(), interaction.GetContact("m_avg_Pelvis").GetForward(), Signals);

			Geometry.Setup(Geometry.Resolution);
			Geometry.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents(), InteractionSmoothing);
			yield return new WaitForSeconds(0f);

			IsInteracting = false;
			Controller.ActiveInteraction = null;
		}
	}


	protected override void OnGUIDerived() {
	}

	protected override void OnRenderObjectDerived() {
		Controller.Draw();


		if(ShowSensorPositions) {
			UltiDraw.Begin();

			for(int k=0; k<2; k++){
				Color[] colors = UltiDraw.GetRainbowColors(BoneNames.Count);

				for(int j=TimeSeries.PivotKey; j<TimeSeries.KeyCount; j++) {
					if (SensorPositions[j][k] != Vector3.zero){
						UltiDraw.DrawSphere(SensorPositions[j][k], Quaternion.identity, 0.05f,  colors[k]);
					}
				}
			}
			// for(int k=0; k<2; k++){
			// 	for(int j=TimeSeries.PivotKey; j<TimeSeries.KeyCount; j++) {
			// 		if (SensorPositions_x[j][k] != Vector3.zero ){
			// 			// Color color =  UltiDraw.Green.Transparent((float)(TimeSeries.KeyCount? -j - TimeSeries.PivotKey) /(float)TimeSeries.KeyCount);
			// 			UltiDraw.DrawSphere(SensorPositions_x[j][k], Quaternion.identity, 0.05f,  UltiDraw.Blue);
			// 		}
			// 	}
			// }
			UltiDraw.End();
		}
		if(ShowContactPositions) {
			UltiDraw.Begin();
			for(int k=0; k<BoneNames.Count; k++){
				Color[] colors = UltiDraw.GetRainbowColors(BoneNames.Count);

				if (ContactPositions[k] != Vector3.zero){
					UltiDraw.DrawSphere(ContactPositions[k], Quaternion.identity, 0.1f, colors[k]);	
				}
			}
			UltiDraw.End();
		}

		// if(ShowIK) {
		// 	UltiDraw.Begin();
		// 	for(int b=0; b<BoneNames.Count; b++){
		// 		UltiDraw.DrawSphere(ik_obj[b], Quaternion.identity, 0.05f, UltiDraw.Blue);	
		// 		UltiDraw.DrawSphere(ik_tar[b], Quaternion.identity, 0.05f, UltiDraw.Red);	
		// 	}
		// 	UltiDraw.End();
		// }

    }

	private float[] GenerateGating() {
		List<float> values = new List<float>();

		for(int k=0; k<TimeSeries.KeyCount; k++) {
			int index = TimeSeries.GetKey(k).Index;
			Vector2 phase = Utility.PhaseVector(PhaseSeries.Values[index]);
			for(int i=0; i<StyleSeries.Styles.Length; i++) {
				float magnitude = StyleSeries.Values[index][i];
				magnitude = Utility.Normalise(magnitude, 0f, 1f, -1f ,1f);
				values.Add(magnitude * phase.x);
				values.Add(magnitude * phase.y);
			}
			for(int i=0; i<GoalSeries.Actions.Length; i++) {
				float magnitude = GoalSeries.Values[index][i];
				magnitude = Utility.Normalise(magnitude, 0f, 1f, -1f ,1f);
				Matrix4x4 root = RootSeries.Transformations[index];
				root[1,3] = 0f;
				Matrix4x4 goal = GoalSeries.Transformations[index];
				goal[1,3] = 0f;
				float distance = Vector3.Distance(root.GetPosition(), goal.GetPosition());
				float angle = Quaternion.Angle(root.GetRotation(), goal.GetRotation());
				values.Add(magnitude * phase.x);
				values.Add(magnitude * phase.y);
				values.Add(magnitude * distance * phase.x);
				values.Add(magnitude * distance * phase.y);
				values.Add(magnitude * angle * phase.x);
				values.Add(magnitude * angle * phase.y);
			}
		}
		return values.ToArray();
	}

	private void ApplyStaticGoal(Vector3 position, Vector3 direction, float[] actions) {
		//Transformations
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			float weight = TimeSeries.GetWeight1byN1(i, 2f);
			float positionBlending = weight * UserControl;
			float directionBlending = weight * UserControl;
			Matrix4x4Extensions.SetPosition(ref GoalSeries.Transformations[i], Vector3.Lerp(GoalSeries.Transformations[i].GetPosition(), position, positionBlending));
			Matrix4x4Extensions.SetRotation(ref GoalSeries.Transformations[i], Quaternion.LookRotation(Vector3.Slerp(GoalSeries.Transformations[i].GetForward(), direction, directionBlending), Vector3.up));
		}

		//Actions
		for(int i=TimeSeries.Pivot; i<TimeSeries.Samples.Length; i++) {
			float w = (float)(i-TimeSeries.Pivot) / (float)(TimeSeries.FutureSampleCount);
			w = Utility.Normalise(w, 0f, 1f, 1f/TimeSeries.FutureKeyCount, 1f);
			for(int j=0; j<GoalSeries.Actions.Length; j++) {
				float weight = GoalSeries.Values[i][j];
				weight = 2f * (0.5f - Mathf.Abs(weight - 0.5f));
				weight = Utility.Normalise(weight, 0f, 1f, UserControl, 1f-UserControl);
				if(actions[j] != GoalSeries.Values[i][j]) {
					GoalSeries.Values[i][j] = Mathf.Lerp(
						GoalSeries.Values[i][j], 
						Mathf.Clamp(GoalSeries.Values[i][j] + weight * UserControl * Mathf.Sign(actions[j] - GoalSeries.Values[i][j]), 0f, 1f),
						w);
				}
			}
		}
	}

	private void ApplyDynamicGoal(Matrix4x4 root, Vector3 move, float turn, float[] actions) {
		//Transformations
		Vector3[] positions_blend = new Vector3[TimeSeries.Samples.Length];
		Vector3[] directions_blend = new Vector3[TimeSeries.Samples.Length];
		float time = 2f;
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			float weight = TimeSeries.GetWeight1byN1(i, 0.5f);
			float bias_pos = 1.0f - Mathf.Pow(1.0f - weight, 0.75f);
			float bias_dir = 1.0f - Mathf.Pow(1.0f - weight, 0.75f);
			directions_blend[i] = Quaternion.AngleAxis(bias_dir * turn, Vector3.up) * Vector3.ProjectOnPlane(root.GetForward(), Vector3.up).normalized;
			if(i==0) {
				positions_blend[i] = root.GetPosition() + 
					Vector3.Lerp(
					GoalSeries.Transformations[i+1].GetPosition() - GoalSeries.Transformations[i].GetPosition(), 
					time / (TimeSeries.Samples.Length - 1f) * (Quaternion.LookRotation(directions_blend[i], Vector3.up) * move),
					bias_pos
					);
			} else {
				positions_blend[i] = positions_blend[i-1] + 
					Vector3.Lerp(
					GoalSeries.Transformations[i].GetPosition() - GoalSeries.Transformations[i-1].GetPosition(), 
					time / (TimeSeries.Samples.Length - 1f) * (Quaternion.LookRotation(directions_blend[i], Vector3.up) * move),
					bias_pos
					);
			}
		}
		for(int i=0; i<TimeSeries.Samples.Length; i++) {
			Matrix4x4Extensions.SetPosition(ref GoalSeries.Transformations[i], Vector3.Lerp(GoalSeries.Transformations[i].GetPosition(), positions_blend[i], UserControl));
			Matrix4x4Extensions.SetRotation(ref GoalSeries.Transformations[i], Quaternion.Slerp(GoalSeries.Transformations[i].GetRotation(), Quaternion.LookRotation(directions_blend[i], Vector3.up), UserControl));
		}
		
		//Actions
		for(int i=TimeSeries.Pivot; i<TimeSeries.Samples.Length; i++) {
			float w = (float)(i-TimeSeries.Pivot) / (float)(TimeSeries.FutureSampleCount);
			w = Utility.Normalise(w, 0f, 1f, 1f/TimeSeries.FutureKeyCount, 1f);
			for(int j=0; j<GoalSeries.Actions.Length; j++) {
				float weight = GoalSeries.Values[i][j];
				weight = 2f * (0.5f - Mathf.Abs(weight - 0.5f));
				weight = Utility.Normalise(weight, 0f, 1f, UserControl, 1f-UserControl);
				if(actions[j] != GoalSeries.Values[i][j]) {
					GoalSeries.Values[i][j] = Mathf.Lerp(
						GoalSeries.Values[i][j], 
						Mathf.Clamp(GoalSeries.Values[i][j] + weight * UserControl * Mathf.Sign(actions[j] - GoalSeries.Values[i][j]), 0f, 1f),
						w);
				}
			}
		}

	}
}
