#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class TimeSeriesModule : Module {

	public int PastKeys = 6;
	public int FutureKeys = 6;
	public float PastWindow = 1f;
	public float FutureWindow = 1f;
	public int Resolution = 1;

	public bool body = false;
	public int additional_key_frame;

	public int[] BoneIndexes;

	public override ID GetID() {
		return ID.TimeSeries;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		
	}


	protected override void DerivedDraw(MotionEditor editor) {
		TimeSeries timeSeries = GetTimeSeries(editor.GetCurrentFrame(), editor.Mirror, 1f/editor.TargetFramerate, body);
		foreach(TimeSeries.Series series in timeSeries.Data) {
			if(series is TimeSeries.Root && Data.GetModule(ID.Root).Visualise) {
				((TimeSeries.Root)series).Draw();
			}
			if(series is TimeSeries.Style && Data.GetModule(ID.Style).Visualise) {
				((TimeSeries.Style)series).Draw();
			}
			if(series is TimeSeries.Goal && Data.GetModule(ID.Goal).Visualise) {
				((TimeSeries.Goal)series).Draw();
			}
			if(series is TimeSeries.Contact && Data.GetModule(ID.Contact).Visualise) {
				((TimeSeries.Contact)series).Draw();
			}
			if(series is TimeSeries.Phase && Data.GetModule(ID.Phase).Visualise) {
				((TimeSeries.Phase)series).Draw();
			}
			if(series is TimeSeries.Hand && Data.GetModule(ID.Hand).Visualise) {
				((TimeSeries.Hand)series).Draw();
			}
		}
	}

	protected override void DerivedInspector(MotionEditor editor) {
		EditorGUILayout.BeginHorizontal();
		GUILayout.FlexibleSpace();
		EditorGUILayout.LabelField("Past Keys", GUILayout.Width(100f));
		PastKeys = EditorGUILayout.IntField(PastKeys, GUILayout.Width(50f));
		EditorGUILayout.LabelField("Future Keys", GUILayout.Width(100f));
		FutureKeys = EditorGUILayout.IntField(FutureKeys, GUILayout.Width(50f));
		EditorGUILayout.LabelField("Past Window", GUILayout.Width(100f));
		PastWindow = EditorGUILayout.FloatField(PastWindow ,GUILayout.Width(50f));
		EditorGUILayout.LabelField("Future Window", GUILayout.Width(100f));
		FutureWindow = EditorGUILayout.FloatField(FutureWindow, GUILayout.Width(50f));
		EditorGUILayout.LabelField("Resolution", GUILayout.Width(100f));
		Resolution = Mathf.Max(EditorGUILayout.IntField(Resolution, GUILayout.Width(50f)), 1);
		GUILayout.FlexibleSpace();
		EditorGUILayout.EndHorizontal();
	}

	public TimeSeries GetTimeSeries(Frame frame, bool mirrored, float delta, bool body_goal) {
		return GetTimeSeries(frame, mirrored, PastKeys, FutureKeys, PastWindow, FutureWindow, Resolution, delta, body_goal);
	}

	public TimeSeries GetTimeSeries(Frame frame, bool mirrored, int pastKeys, int futureKeys, float pastWindow, float futureWindow, int resolution, float delta, bool body_goal) {
		TimeSeries timeSeries = new TimeSeries(pastKeys, futureKeys, pastWindow, futureWindow, resolution);
		// var BoneNames = new List<string> {"m_avg_R_Wrist", "m_avg_L_Wrist", "m_avg_R_Ankle", "m_avg_L_Ankle"};
		var BoneNames = new List<string> {"m_avg_R_Wrist", "m_avg_L_Wrist"};

		// int[] BoneIndexes = new int[] {27, 21};
		foreach(Module module in Data.Modules) {
			if(module is RootModule) {
				RootModule m = (RootModule)module;
				TimeSeries.Root series = new TimeSeries.Root(timeSeries);
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					if(t < 0f || t > Data.GetTotalTime()) {
						series.Transformations[i] = m.GetEstimatedRootTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored);
						series.Velocities[i] = m.GetEstimatedRootVelocity(frame, timeSeries.Samples[i].Timestamp, mirrored, delta);
					} else {
						series.Transformations[i] = m.GetRootTransformation(Data.GetFrame(t), mirrored);
						series.Velocities[i] = m.GetRootVelocity(Data.GetFrame(t), mirrored, delta);
					}
				}
			}
			if(module is StyleModule) {
				StyleModule m = (StyleModule)module;
				TimeSeries.Style series = new TimeSeries.Style(timeSeries, m.GetNames());
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					series.Values[i] = m.GetStyles(Data.GetFrame(t));
				}
			}
			if(module is GoalModule) {
				GoalModule m = (GoalModule)module;
				TimeSeries.Goal series = new TimeSeries.Goal(timeSeries, m.GetNames());
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;		
					series.Transformations[i] = m.Target.GetGoalTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored, delta, body_goal);
					series.Values[i] = m.GetActions(Data.GetFrame(t), delta);
					
				}
			}
			if(module is ContactModule) {
				ContactModule m = (ContactModule)module;
				TimeSeries.Contact series = new TimeSeries.Contact(timeSeries, m.GetNames());

				// copying goal contact 
				GoalModule m_goal = (GoalModule)Data.GetModule(Module.ID.Goal);
				TimeSeries.Goal series_goal = new TimeSeries.Goal(timeSeries, m_goal.GetNames());

				
				// Debug.Log(corrected.Length);
				// Debug.Log(series.CorrectedTransformations.Length);
				for(int j=0; j<BoneNames.Count; j++){
					m.GetProcessedContacts(mirrored, BoneNames[j]);
				}
				for(int j=0; j<BoneNames.Count; j++){
					for(int i=0; i<timeSeries.Samples.Length; i++) {
						// Debug.Log(string.Format("{0} {1} {2}", frame.Timestamp,timeSeries.Samples[i].Timestamp, timeSeries.Samples[i+1].Timestamp));

						float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
						// Debug.Log(string.Format("{0}", processed_contacts[Data.GetFrame(t+1f/30f).Index]));
						// Debug.Log(string.Format("{0}", Data.GetFrame(t).Index));
						series.Values[i] = m.GetContacts(Data.GetFrame(t), mirrored);


						// series.Transformations[i][j] = Matrix4x4.TRS(m.GetClusteredContact(frame, mirrored, BoneNames[j]), 
						// 											m_goal.Target.GetGoalTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored, delta, body_goal).GetRotation(),
						// 											Vector3.one);
						series.Transformations[i][j] = Matrix4x4.TRS(m.GetCorrectedGoalPoint(Data.GetFrame(t), mirrored, BoneNames[j]), 
																     m_goal.Target.GetGoalTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored, delta, body_goal).GetRotation(),
																	 Vector3.one);


						// Debug.Log(string.Format("{0} {1}",series.CorrectedTransformations[i][j].GetType(),  m.GetCurrentHandTarget(Data.GetFrame(t), mirrored, BoneNames[j]).GetType()));
						series.CorrectedTransformations[i][j] = m.GetCurrentHandTarget(Data.GetFrame(t), mirrored, BoneNames[j]);
						
						// Debug.Log(string.Format("{0} {1}", Data.GetFrame(t).Index, Data.GetFrame(t+1f/30f).Index));

						series.ValuesPred[i] = m.GetContacts(Data.GetFrame(t+(1f/30f)), mirrored);
						// Debug.Log(string.Format("1 {0} {1}", Data.GetFrame(t+(1f/30f)).Index, Data.GetFrame(t).Index+2));

						series.TransformationsPred[i][j] = Matrix4x4.TRS(m.GetCorrectedGoalPoint(Data.GetFrame(t+1f/30f), mirrored, BoneNames[j]), 
																		 m_goal.Target.GetGoalTransformation(Data.GetFrame(frame.Timestamp+1f/30f), timeSeries.Samples[i].Timestamp, mirrored, delta, body_goal).GetRotation(),
																		 Vector3.one);
						// series.TransformationsPred[i][j] = Matrix4x4.TRS(m.GetClusteredContact(Data.GetFrame(t+(1f/30f)), mirrored, BoneNames[j]), 
						// 												m_goal.Target.GetGoalTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored, delta, body_goal).GetRotation(),
						// 												Vector3.one);

																		
						series.CorrectedTransformationsPred[i][j] = m.GetCurrentHandTarget(Data.GetFrame(t+1f/30f), mirrored, BoneNames[j]);
						series.Speed[i][j] = (series.CorrectedTransformationsPred[i][j].GetPosition() - series.CorrectedTransformations[i][j].GetPosition()).magnitude;
						
						// local phase
						series.LocalPhases[i][j] = m.GetLocalPhases(Data.GetFrame(t), mirrored, BoneNames[j]);


					}
				}
			}

			if(module is PhaseModule) {
				PhaseModule m = (PhaseModule)module;
				TimeSeries.Phase series = new TimeSeries.Phase(timeSeries);
				for(int i=0; i<timeSeries.Samples.Length; i++) {
					float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;
					series.Values[i] = m.GetPhase(Data.GetFrame(t), mirrored);
					// for(int c=0; c<5; c++) { 
					// 	series.LocalPhaseVectors[i][c] = m.GetLocalPhaseVector(Data.GetFrame(t), mirrored, c);
					// 	series.Amplitudes[i][c]  = m.GetAmplitude(Data.GetFrame(t), mirrored, c);
					// 	series.Frequencies[i][c]  = m.GetFrequency(Data.GetFrame(t), mirrored, c);
					// 	// Debug.Log(string.Format("{0} {1}", i, series.LocalPhaseVectors[i][c]) );

					// }
				}
			}

		
		// 	if(module is HandModule) {
		// 		HandModule m = (HandModule)module;
		// 		TimeSeries.Hand series = new TimeSeries.Hand(timeSeries);

		// 		ContactModule m_contact = (ContactModule)Data.GetModule(Module.ID.Contact);
		// 		TimeSeries.Contact series_contact = new TimeSeries.Contact(timeSeries, m_contact.GetNames());
		// 		for(int j=0; j<2; j++) {
		// 			for(int i=0; i<timeSeries.Samples.Length; i++) {
		// 				float t = frame.Timestamp + timeSeries.Samples[i].Timestamp;		
		// 				if(t < 0f || t > Data.GetTotalTime()) {
		// 					series.Transformations[i][j] = m.GetEstimatedTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored, BoneNames[j]);
		// 					series.InitialTransformations[i][j] = m.GetEstimatedTransformation(frame, timeSeries.Samples[i].Timestamp, mirrored, BoneNames[j]);
		// 					series.Velocities[i][j] = m.GetEstimatedVelocity(frame, timeSeries.Samples[i].Timestamp, mirrored, delta, BoneNames[j]);
		// 				} 
		// 				else {
		// 					series.Transformations[i][j] = m.GetTransformation(Data.GetFrame(t), mirrored, BoneNames[j]);
		// 					series.InitialTransformations[i][j] = m.GetTransformation(Data.GetFrame(t), mirrored, BoneNames[j]);
		// 					series.Velocities[i][j] = m.GetVelocity(Data.GetFrame(t), mirrored, delta, BoneNames[j]);
		// 					// m_contatc.GetTargets
		// 				}	
		// 			}
		// 		}
		// 	}
		
		}



		
		return timeSeries;
	}

}
#endif
