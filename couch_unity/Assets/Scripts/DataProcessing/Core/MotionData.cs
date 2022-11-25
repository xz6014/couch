#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;

public class MotionData : ScriptableObject {

	public Hierarchy Source = null;
	public Frame[] Frames = new Frame[0];
	public Module[] Modules = new Module[0];
	public Sequence[] Sequences = new Sequence[0];

	public float Framerate = 1f;
	public Vector3 Offset = Vector3.zero;
	public float Scale = 1f;
	// Axis of reference
	public Axis MirrorAxis = Axis.XPositive;
	public int[] Symmetry = new int[0];

	public bool Export = true;
	public bool Symmetric = true;

	public void InspectAll(bool value) {
		foreach(Module module in Modules) {
			module.Inspect = value;
		}
	}

	public void VisualiseAll(bool value) {
		foreach(Module module in Modules) {
			module.Visualise = value;
		}
	}
	

	public string GetName() {
		return name;
	}
	

	public float GetTotalTime() {
		return Frames.Length / Framerate;
	}

	public int GetTotalFrames() {
		return Frames.Length;
	}

	public Frame GetFrame(int index) {
		return Frames[Mathf.Clamp(index-1, 0, Frames.Length-1)];
		/*
		if(index < 1 || index > GetTotalFrames()) {
			Debug.Log("Please specify an index between 1 and " + GetTotalFrames() + ". Given " + index + ".");
			return null;
		}
		return Frames[index-1];
		*/
	}

	public Frame GetFrame(float time) {
		return Frames[Mathf.Clamp(Mathf.RoundToInt(time * Framerate), 0, Frames.Length-1)];
		/*
		if(time < 0f || time > GetTotalTime()) {
			Debug.Log("Please specify a time between 0 and " + GetTotalTime() + ". Given " + time + ".");
			return null;
		}
		return GetFrame(Mathf.Min(Mathf.RoundToInt(time * Framerate) + 1, GetTotalFrames()));
		*/
	}

	public Frame[] GetFrames(int start, int end) {
		if(start < 1 || end > GetTotalFrames()) {
			Debug.Log("Please specify indices between 1 and " + GetTotalFrames() + ". Given " + start + " and " + end + ".");
			return null;
		}
		int count = end-start+1;
		Frame[] frames = new Frame[count];
		for(int i=start; i<=end; i++) {
			frames[i-start] = Frames[i-1];
		}
		return frames;
	}

	public Frame[] GetFrames(float start, float end) {
		if(start < 0f || end > GetTotalTime()) {
			Debug.Log("Please specify times between 0 and " + GetTotalTime() + ". Given " + start + " and " + end + ".");
			return null;
		}
		return GetFrames(GetFrame(start).Index, GetFrame(end).Index);
	}

	public void AddSequence() {
		ArrayExtensions.Add(ref Sequences, new Sequence(1, GetTotalFrames()));
	}

	public void AddSpecificSequence(int start, int end) {
		ArrayExtensions.Add(ref Sequences, new Sequence(start, end));
	}


	public void RemoveSequence(Sequence sequence) {
		ArrayExtensions.Remove(ref Sequences, sequence);
	}

	public bool ContainedInSequences(Frame frame) {
		foreach(Sequence s in Sequences) {
			if(s.Contains(frame.Index)) {
				return true;
			}
		}
		return false;
	}

	public Sequence GetUnrolledSequence() {
		// if(Sequences.Length == 0) {
		// 	return new Sequence(1, Frames.Length);
		// }
		if(Sequences.Length == 1) {
			return Sequences[0];
		}
		int start = 1;
		int end = Frames.Length;
		foreach(Sequence seq in Sequences) {
			start = Mathf.Min(seq.Start, start);
			end = Mathf.Max(seq.End, end);
		}
		return new Sequence(start, end);
	}

	public void Load() {
		/*
		//Reimport from deprecated format
		if(name != "Data") {
			Name = name;
			Debug.Log("Reimporting file " + name + " from deprecated format.");
			AssetDatabase.CreateFolder(Path.GetDirectoryName(AssetDatabase.GetAssetPath(this)), name);
			AssetDatabase.MoveAsset(AssetDatabase.GetAssetPath(this), Path.GetDirectoryName(AssetDatabase.GetAssetPath(this)) + "/" + name + "/Data.asset");
		}
		//Setup scenes and directories
		if(!Directory.Exists(Path.GetDirectoryName(AssetDatabase.GetAssetPath(this))+"/Scenes")) {
			Debug.Log("Setting up scenes for data " + GetName() + ". No scene folder found.");
			AssetDatabase.CreateFolder(Path.GetDirectoryName(AssetDatabase.GetAssetPath(this)), "Scenes");
			Scenes = new Scene[0];
			AddScene();
		} else {
			string[] files = Directory.GetFiles(Path.GetDirectoryName(AssetDatabase.GetAssetPath(this))+"/Scenes");
			foreach(string file in files) {
				if(file.Contains(".meta")) {
					ArrayExtensions.Remove(ref files, file);
				}
			}
			if(Scenes.Length != files.Length) {
				Debug.Log("Found " + files.Length + " files for " + Scenes.Length + " scenes for data " + GetName() + ".");
				if(files.Length == 0) {
					Scenes = new Scene[0];
					AddScene();
					Debug.Log("Created new scene at index 1.");
				} else {
					Scene[] scenes = new Scene[files.Length];
					for(int i=0; i<scenes.Length; i++) {
						string file = files[i].Substring(files[i].LastIndexOf("/")+1).Replace(".prefab", "");
						int index = int.Parse(file.Substring(5));
						scenes[i] = System.Array.Find(Scenes, x => x.Index == index);
						if(scenes[i] == null) {
							scenes[i] = new Scene(this);
							Debug.Log("Created new scene at index " + (i+1) + ".");
						} else {
							Debug.Log("Reassigned scene at index " + (i+1) + " from index " + index + ".");
						}
					}
					for(int i=0; i<scenes.Length; i++) {
						scenes[i].Index = i+1;
						string oldPath = files[i];
						string newPath = files[i].Substring(0, files[i].LastIndexOf("/")+1) + "Scene" + scenes[i].Index + ".prefab";
						AssetDatabase.MoveAsset(oldPath, newPath);
						Debug.Log("Moved scene " + oldPath + " to " + newPath + ".");
					}
					Scenes = scenes;
				}
				AssignActiveScene();
			}
		}
		*/

		//Check Naming
		if(name == "Data") {
			Debug.Log("Updating name of asset at " + AssetDatabase.GetAssetPath(this) + ".");
			string dataName = Directory.GetParent(AssetDatabase.GetAssetPath(this)).Name;
			int dataDot = dataName.LastIndexOf(".");
			if(dataDot != -1) {
				dataName = dataName.Substring(0, dataDot);
			}
			AssetDatabase.RenameAsset(AssetDatabase.GetAssetPath(this), dataName);
			string parentDirectory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(this));
			int parentDot = parentDirectory.LastIndexOf(".");
			if(parentDot != -1) {
				AssetDatabase.MoveAsset(parentDirectory, parentDirectory.Substring(0, parentDot));
			}
		}

		//Check modules
		for(int i=0; i<Modules.Length; i++) {
			if(Modules[i] == null) {
				Debug.Log("Removing missing module in " + GetName() + ".");
				ArrayExtensions.RemoveAt(ref Modules, i);
				i--;
			}
		}

		//Open Scene
		GetScene();
	}

	public void Unload() {
		Scene scene = EditorSceneManager.GetSceneByName(name);
		if(Application.isPlaying) {
			SceneManager.UnloadSceneAsync(scene);
		} else {
			EditorSceneManager.CloseScene(scene, false);
			EditorCoroutines.StartCoroutine(RemoveScene(scene), this);
		}
	}

	private IEnumerator RemoveScene(Scene scene) {
		yield return new WaitForSeconds(1f);
		EditorSceneManager.CloseScene(scene, true);
		yield return new WaitForSeconds(0f);
		EditorApplication.RepaintHierarchyWindow();
	}

	public void Save() {
		if(!Application.isPlaying) {
			EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetSceneByName(name));
			EditorSceneManager.SaveScene(EditorSceneManager.GetSceneByName(name));
			EditorUtility.SetDirty(this);
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
		}
	}

	public void ChangePositionsAndSave(string source_folder){
		// if(!Application.isPlaying) {

			DirectoryInfo info = new DirectoryInfo(source_folder);
			FileInfo[] items = info.GetFiles("*.bvh");

			for(int f=0; f<items.Length; f++) {
				string fileName = items[f].Name.Replace(".bvh", "");

				if (fileName == GetName()){
				// Debug.Log(fileName);
				// Debug.Log(GetName());
					// if(!Directory.Exists(destination+"/"+fileName) ) {
					// AssetDatabase.CreateFolder(destination_folder, fileName);
					MotionData data = ScriptableObject.CreateInstance<MotionData>();
					data.name = fileName;
					// AssetDatabase.CreateAsset(this, destination_folder+"/"+fileName+"/"+data.name+".asset");
					// }
				
					Debug.Log(source_folder + "/" + fileName + ".bvh");

					LoadBVH(source_folder + "/" + fileName + ".bvh", data);
					// Debug.Log(Frames[0].World.Length);
					// Debug.Log(data.GetFrame(0).World.Length);
					for(int k=0; k<Frames.Length; k++) {
						Frames[k].World = (Matrix4x4[])data.GetFrame(k).World.Clone();
					}
					EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetSceneByName(name));
					EditorSceneManager.SaveScene(EditorSceneManager.GetSceneByName(name));
					EditorUtility.SetDirty(this);
					AssetDatabase.SaveAssets();
					AssetDatabase.Refresh();
					Debug.Log("Replaced World Transformations");

				}
			}
	}


	private void LoadBVH(string filename, MotionData data){
		//Create Source Data
		// MotionData data = ScriptableObject.CreateInstance<MotionData>();
		// data.name = filename;
		// string fileName = Files[f].Object.Name.Replace(".bvh", "");


		string[] lines = System.IO.File.ReadAllLines(filename);
		char[] whitespace = new char[] {' '};
		int index = 0;

		List<Vector3> offsets = new List<Vector3>();
		List<int[]> channels = new List<int[]>();
		List<float[]> motions = new List<float[]>();
		data.Source = new MotionData.Hierarchy();
		string name = string.Empty;
		string parent = string.Empty;
		Vector3 offset = Vector3.zero;
		int[] channel = null;

		for(index = 0; index<lines.Length; index++) {
			if(lines[index] == "MOTION") {
				break;
			}
			string[] entries = lines[index].Split(whitespace);
			for(int entry=0; entry<entries.Length; entry++) {
				if(entries[entry].Contains("ROOT")) {
					parent = "None";
					name = entries[entry+1];
					break;
				} else if(entries[entry].Contains("JOINT")) {
					parent = name;
					name = entries[entry+1];
					break;
				} else if(entries[entry].Contains("End")) {
					parent = name;
					name = name+entries[entry+1];
					string[] subEntries = lines[index+2].Split(whitespace);
					for(int subEntry=0; subEntry<subEntries.Length; subEntry++) {
						if(subEntries[subEntry].Contains("OFFSET")) {
							offset.x = FileUtility.ReadFloat(subEntries[subEntry+1]);
							offset.y = FileUtility.ReadFloat(subEntries[subEntry+2]);
							offset.z = FileUtility.ReadFloat(subEntries[subEntry+3]);
							break;
						}
					}
					data.Source.AddBone(name, parent);
					offsets.Add(offset);
					channels.Add(new int[0]);
					index += 2;
					break;
				} else if(entries[entry].Contains("OFFSET")) {
					// Debug.Log(entries[entry+1]);
					offset.x = FileUtility.ReadFloat(entries[entry+1]);
					offset.y = FileUtility.ReadFloat(entries[entry+2]);
					offset.z = FileUtility.ReadFloat(entries[entry+3]);
					break;
				} else if(entries[entry].Contains("CHANNELS")) {
					channel = new int[FileUtility.ReadInt(entries[entry+1])];
					for(int i=0; i<channel.Length; i++) {
						if(entries[entry+2+i] == "Xposition") {
							channel[i] = 1;
						} else if(entries[entry+2+i] == "Yposition") {
							channel[i] = 2;
						} else if(entries[entry+2+i] == "Zposition") {
							channel[i] = 3;
						} else if(entries[entry+2+i] == "Xrotation") {
							channel[i] = 4;
						} else if(entries[entry+2+i] == "Yrotation") {
							channel[i] = 5;
						} else if(entries[entry+2+i] == "Zrotation") {
							channel[i] = 6;
						}
					}
					data.Source.AddBone(name, parent);
					offsets.Add(offset);
					channels.Add(channel);
					break;
				} else if(entries[entry].Contains("}")) {
					name = parent;
					parent = name == "None" ? "None" : data.Source.FindBone(name).Parent;
					break;
				}
			}
		}

		//Set Frames
		index += 1;
		while(lines[index].Length == 0) {
			index += 1;
		}
		ArrayExtensions.Resize(ref data.Frames, FileUtility.ReadInt(lines[index].Substring(8)));

		//Set Framerate
		index += 1;
		data.Framerate = Mathf.RoundToInt(1f / FileUtility.ReadFloat(lines[index].Substring(12)));

		//Compute Frames
		index += 1;
		for(int i=index; i<lines.Length; i++) {
			motions.Add(FileUtility.ReadArray(lines[i]));
		}
		for(int k=0; k<data.GetTotalFrames(); k++) {
			data.Frames[k] = new Frame(data, k+1, (float)k / data.Framerate);
			int idx = 0;
			for(int i=0; i<data.Source.Bones.Length; i++) {
				MotionData.Hierarchy.Bone info = data.Source.Bones[i];
				Vector3 position = Vector3.zero;
				Quaternion rotation = Quaternion.identity;
				for(int j=0; j<channels[i].Length; j++) {
					if(channels[i][j] == 1) {
						position.x = motions[k][idx]; idx += 1;
					}
					if(channels[i][j] == 2) {
						position.y = motions[k][idx]; idx += 1;
					}
					if(channels[i][j] == 3) {
						position.z = motions[k][idx]; idx += 1;
					}
					if(channels[i][j] == 4) {
						rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.right); idx += 1;
					}
					if(channels[i][j] == 5) {
						rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.up); idx += 1;
					}
					if(channels[i][j] == 6) {
						rotation *= Quaternion.AngleAxis(motions[k][idx], Vector3.forward); idx += 1;
					}
				}

				position = (position == Vector3.zero ? offsets[i] : position) ; //unit scale
				Matrix4x4 local = Matrix4x4.TRS(position, rotation, Vector3.one);
				// if(Flip) {
				// 	local = local.GetMirror(Axis);
				// }
				data.Frames[k].World[i] = info.Parent == "None" ? local : data.Frames[k].World[data.Source.FindBone(info.Parent).Index] * local;
			}
			/*
			for(int i=0; i<data.Source.Bones.Length; i++) {
				data.Frames[k].Local[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(data.Corrections[i]), Vector3.one);
				data.Frames[k].World[i] *= Matrix4x4.TRS(Vector3.zero, Quaternion.Euler(data.Corrections[i]), Vector3.one);
			}
			*/
		}

		if(data.GetTotalFrames() == 1) {
			Frame reference = data.Frames.First();
			ArrayExtensions.Resize(ref data.Frames, Mathf.RoundToInt(data.Framerate));
			for(int k=0; k<data.GetTotalFrames(); k++) {
				data.Frames[k] = new Frame(data, k+1, (float)k / data.Framerate);
				data.Frames[k].World = (Matrix4x4[])reference.World.Clone();
			}
		}
		
	}


	public Scene GetScene() {
		for(int i=0; i<SceneManager.sceneCount; i++) {
			if(SceneManager.GetSceneAt(i).name == name) {
				return SceneManager.GetSceneAt(i);
			}
		}
		if(Application.isPlaying) {

			if(File.Exists(GetScenePath())) {
				EditorSceneManager.LoadSceneInPlayMode(GetScenePath(), new LoadSceneParameters(LoadSceneMode.Additive));
			} else {
				Debug.Log("Creating temporary scene for data " + name + ".");
				SceneManager.CreateScene(name);			}
		} else {
			Scene active = EditorSceneManager.GetActiveScene();
			if(File.Exists(GetScenePath())) {
				EditorSceneManager.OpenScene(GetScenePath(), OpenSceneMode.Additive);
			} else {
				Debug.Log("Recreating scene for data " + name + ".");
				EditorSceneManager.SaveScene(EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive), GetScenePath());
			}
			EditorSceneManager.SetActiveScene(SceneManager.GetSceneByName(name));
			Lightmapping.bakedGI = false;
			Lightmapping.realtimeGI = false;
			EditorSceneManager.SetActiveScene(active);
		}
		return SceneManager.GetSceneByName(name);
	}

	private string GetScenePath() {
		return Path.GetDirectoryName(AssetDatabase.GetAssetPath(this)) + "/" + name + ".unity";
	}

	public void CreateScene() {
		UnityEngine.SceneManagement.Scene active = EditorSceneManager.GetActiveScene();
		UnityEngine.SceneManagement.Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
		EditorSceneManager.SetActiveScene(scene);
		Lightmapping.bakedGI = false;
		Lightmapping.realtimeGI = false;
		EditorSceneManager.SetActiveScene(active);
		Debug.Log(GetScenePath());
		EditorSceneManager.SaveScene(scene, GetScenePath());
		EditorSceneManager.CloseScene(scene, true);
	}

	public Actor CreateActor() {
		Actor actor = new GameObject("Skeleton").AddComponent<Actor>();
		List<Transform> instances = new List<Transform>();
		for(int i=0; i<Source.Bones.Length; i++) {
			Transform instance = new GameObject(Source.Bones[i].Name).transform;
			instance.SetParent(Source.Bones[i].Parent == "None" ? actor.GetRoot() : actor.FindTransform(Source.Bones[i].Parent));
			Matrix4x4 matrix = Frames.First().GetBoneTransformation(i, false);
			instance.position = matrix.GetPosition();
			instance.rotation = matrix.GetRotation();
			instance.localScale = Vector3.one;
			instances.Add(instance);
		}
		Transform root = actor.FindTransform(Source.Bones[0].Name);
		root.position = new Vector3(0f, root.position.y, 0f);
		root.rotation = Quaternion.Euler(root.eulerAngles.x, 0f, root.eulerAngles.z);
		actor.ExtractSkeleton(instances.ToArray());
		return actor;
	}

	public Module AddModule(Module.ID type) {
		Module module = System.Array.Find(Modules, x => x.GetID() == type);
		if(module != null) {
			// Debug.Log("Module of type " + type + " already exists in " + GetName() + ".");
		} else {
			string id = type + "Module";
			module = (Module)ScriptableObject.CreateInstance(id);
			if(module == null) {
				Debug.Log("Module of class type " + id + " could not be loaded in " + GetName() + ".");
			} else {
				ArrayExtensions.Add(ref Modules, module.Initialise(this));
				AssetDatabase.AddObjectToAsset(Modules.Last(), this);
			}
		}
		return module;
	}

	public void RemoveModule(Module.ID type) {
		Module module = GetModule(type);
		if(!module) {
			Debug.Log("Module of type " + type + " does not exist in " + GetName() + ".");
		} else {
			ArrayExtensions.Remove(ref Modules, module);
			Utility.Destroy(module);
		}
	}

	public void RemoveAllModules() {
		foreach(Module module in Modules) {
			ArrayExtensions.Remove(ref Modules, module);
			Utility.Destroy(module);
		}
	}

	public Module GetModule(Module.ID type) {
		for(int i=0; i<Modules.Length; i++) {
			if(Modules[i].GetID() == type) {
				return Modules[i];
			}
		}
		//Debug.Log("Module of type " + type + " does not exist in " + GetName() + ".");
		return null;
	}

	public Matrix4x4[] SamplePosture(float timestamp) {
		Matrix4x4[] transformations = new Matrix4x4[Source.Bones.Length];
		Frame current = GetFrame(timestamp);
		if(timestamp < current.Timestamp) {
			Frame previous = GetFrame(current.Index-1);
			float ratio = (timestamp - previous.Timestamp) / (current.Timestamp - previous.Timestamp);
			for(int i=0; i<transformations.Length; i++) {
				transformations[i] = Utility.Interpolate(previous.World[i], current.World[i], ratio);
			}
		}
		if(timestamp > current.Timestamp) {
			Frame next = GetFrame(current.Index+1);
			float ratio = (timestamp - current.Timestamp) / (next.Timestamp - current.Timestamp);
			for(int i=0; i<transformations.Length; i++) {
				transformations[i] = Utility.Interpolate(current.World[i], next.World[i], ratio);
			}
		}
		if(timestamp == current.Timestamp) {
			for(int i=0; i<transformations.Length; i++) {
				transformations[i] = current.World[i];
			}
		}
		return transformations;
	}

	public void ResampleMotion(MotionData reference) {
		if(Source.Bones.Length != reference.Source.Bones.Length) {
			Debug.Log("Could not resample motion because number of bones does not match.");
			return;
		}
		foreach(Frame frame in Frames) {
			frame.World = reference.SamplePosture(Utility.Normalise(frame.Timestamp, 0f, Frames.Last().Timestamp, 0f, reference.Frames.Last().Timestamp));
		}
	}

	public void DetectSymmetry() {
		Symmetry = new int[Source.Bones.Length];
		for(int i=0; i<Source.Bones.Length; i++) {
			if(Source.Bones[i].Name.Contains("m_avg_L")) {
				int pivot = Source.Bones[i].Name.IndexOf("m_avg_L");
				Hierarchy.Bone bone = Source.FindBone(Source.Bones[i].Name.Substring(0, pivot)+"m_avg_R"+Source.Bones[i].Name.Substring(pivot+7));
				// Debug.Log(string.Format("{0}: {1}", Source.Bones[i].Name, bone.Name));

				// Debug.Log(Source.Bones[i].Name.Substring(0, pivot)+"m_avg_R"+Source.Bones[i].Name.Substring(pivot+7));
				if(bone == null) {
					Debug.Log("Could not find mapping for " + Source.Bones[i].Name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(Source.Bones[i].Name.Contains("m_avg_R")) {


				int pivot = Source.Bones[i].Name.IndexOf("m_avg_R");
				// Debug.Log(Source.Bones[i].Name.Substring(0, pivot)+"m_avg_R"+Source.Bones[i].Name.Substring(pivot+7));

				Hierarchy.Bone bone = Source.FindBone(Source.Bones[i].Name.Substring(0, pivot)+"m_avg_L"+Source.Bones[i].Name.Substring(pivot+7));
				// Debug.Log(string.Format("{0}: {1}", Source.Bones[i].Name, bone.Name));

				if(bone == null) {
					Debug.Log("Could not find mapping for " + Source.Bones[i].Name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(Source.Bones[i].Name.StartsWith("m_avg_L") && char.IsUpper(Source.Bones[i].Name[1])) {

				Hierarchy.Bone bone = Source.FindBone("m_avg_R"+Source.Bones[i].Name.Substring(1));
				// Debug.Log(string.Format("{0}: {1}", Source.Bones[i].Name, bone.Name));


				if(bone == null) {
					Debug.Log("Could not find mapping for " + Source.Bones[i].Name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else if(Source.Bones[i].Name.StartsWith("m_avg_R") && char.IsUpper(Source.Bones[i].Name[1])) {

				Hierarchy.Bone bone = Source.FindBone("m_avg_L"+Source.Bones[i].Name.Substring(1));
				// Debug.Log(string.Format("{0}: {1}", Source.Bones[i].Name, bone.Name));

				if(bone == null) {
					Debug.Log("Could not find mapping for " + Source.Bones[i].Name + ".");
				} else {
					Symmetry[i] = bone.Index;
				}
			} else {
				Debug.Log(string.Format("No Symmetry detected for bone {0}", Source.Bones[i].Name));

				Symmetry[i] = i;
			}
		}
	}

	public void SetSymmetry(int source, int target) {
		if(Symmetry[source] != target) {
			Symmetry[source] = target;
		}
	}

	[System.Serializable]
	public class Hierarchy {
		public Bone[] Bones;

		private string[] Names = null;

		public Hierarchy() {
			Bones = new Bone[0];
		}

		public void AddBone(string name, string parent) {
			ArrayExtensions.Add(ref Bones, new Bone(Bones.Length, name, parent));
		}

		public Bone FindBone(string name) {
			return System.Array.Find(Bones, x => x.Name == name);
		}

		public Bone FindBoneContains(string name) {
			return System.Array.Find(Bones, x => x.Name.Contains(name));
		}

		public string[] GetBoneNames() {
			if(Names == null || Names.Length != Bones.Length) {
				Names = new string[Bones.Length];
				for(int i=0; i<Bones.Length; i++) {
					Names[i] = Bones[i].Name;
				}
			}
			return Names;
		}

		public bool[] GetBoneFlags(params string[] bones) {
			bool[] flags = new bool[Bones.Length];
			for(int i=0; i<bones.Length; i++) {
				Bone bone = FindBone(bones[i]);
				if(bone != null) {
					flags[bone.Index] = true;
				}
			}
			return flags;
		}

		public int[] GetBoneIndices(params string[] bones) {
			int[] indices = new int[bones.Length];
			for(int i=0; i<bones.Length; i++) {
				Bone bone = FindBone(bones[i]);
				indices[i] = bone == null ? -1 : bone.Index;
			}
			return indices;
		}

		[System.Serializable]
		public class Bone {
			public int Index = -1;
			public string Name = "";
			public string Parent = "";
			public float Mass = 1f;
			public Vector3 Alignment = Vector3.zero;
			public Bone(int index, string name, string parent) {
				Index = index;
				Name = name;
				Parent = parent;
				Mass = 1f;
				Alignment = Vector3.zero;
			}
		}
	}
}
#endif