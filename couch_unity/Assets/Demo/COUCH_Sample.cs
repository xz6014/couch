
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DeepLearning;
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using UnityEditorInternal;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

[ExecuteInEditMode]
public class COUCH_Sample: MonoBehaviour {
	

	public NeuralAnimation Controller;	

	private Frame frame;

	private MotionData[] Files = new MotionData[0];

	// public int ControllerIndex=-1;

	private int CurrentAnimationFrameCount=0;

	private int TransparentFrameCount=40;

	private int fade_num_frame = 40;

    public ContactNetwork ContactNet;

	private bool pause=false;

	private int pause_count;

	private GameObject Chair;

    private Vector3[] contacts;


	public void Start(){
		Controller.enabled = true;
		
		ContactNet.LoadDerived();
		// AddChair();
		contacts = new Vector3[2];

		
	}


	public void Update(){

		if (Application.isPlaying){
			// initializing 
			CurrentAnimationFrameCount ++;
			if (CurrentAnimationFrameCount == 3){
				Controller.enabled = false;
		
				SceneManager.SetActiveScene(SceneManager.GetSceneAt(1));
				GameObject scene = GameObject.Find("session1_chair1(Clone)");
				for(int k = 0; k < scene.transform.transform.childCount; k++)
				{
					GameObject tmp = scene.transform.transform.GetChild(k).gameObject;
					if(tmp.GetComponent<Interaction>() != null & tmp.activeSelf){

						Chair = tmp;
					}
				}

				SceneManager.SetActiveScene(SceneManager.GetSceneAt(0));

			}



			OnPause();
			if (pause==true){
				TransparentFrameCount = 0;	


				List<Vector3> LeftContacts = new List<Vector3>() ;
				List<Vector3> RightContacts = new List<Vector3>();

				Controller.enabled = true;
				RightContacts.Add(contacts[0]);
				LeftContacts.Add(contacts[1]);
					


				Controller.gameObject.GetComponent<COUCH_testing_subject_starting> ().RightContacts = RightContacts;
				Controller.gameObject.GetComponent<COUCH_testing_subject_starting> ().LeftContacts = LeftContacts;
					

				
			

			}
			else{
				if (CurrentAnimationFrameCount > 3 ){
					Inference();
				}
			}


		}


	}
	public void OnPause(){
		
		if (Input.GetKeyDown(KeyCode.N)){
			pause_count ++;
			print("N was pressed");
			// ControllerIndex = pause_count /2;

		}
		if (pause_count % 2 == 0){
			pause = false;
			// if (ControllerIndex > 0){
			// ChangeTexture();
			// }
			Application.targetFrameRate = 1;
		}
		else{
			pause = true;
			Application.targetFrameRate = 1;
		}

	}




    private void Inference() {


        Interaction interaction = Chair.GetComponent<Interaction>();

        if (interaction != null) {
                Vector3[] tmp = ContactNet.PredictGoal(interaction, null);
                
                // Taking care of zeros
                for (int j = 0; j < tmp.Length; j++) {   
                    if (tmp[j].y < 0.1f){
                        tmp[j] = Vector3.zero;
                        // tmp[j].y = -1;
                    }
                }



                for (int j = 0; j < tmp.Length; j++) {   

                    BoxCollider[] voxels = Chair.GetComponentInChildren<VoxelCollider>().GetVoxels();
                    float minDistance = 0.25f;
                    Vector3 nearest = Vector3.zero;
                    // scan all vertices to find nearest
                    foreach (BoxCollider voxel in voxels)
                    {    

                        Vector3 diff = tmp[j] - 0.5f * (voxel.bounds.min + voxel.bounds.max);
                        float dist = diff.magnitude;
                        if (dist < minDistance)
                        {
                            minDistance = dist;
                            nearest =  0.5f * (voxel.bounds.min + voxel.bounds.max);
                        }
                    }
                    if (nearest.y < 0.1f){
                        nearest = Vector3.zero;
                        // nearest.y = -1;

                    }
                
                    contacts[j] = nearest;

                }


        }


    }

	void OnRenderObject(){
		if (Application.isPlaying){
			if (TransparentFrameCount == fade_num_frame | pause == true){
				UltiDraw.Begin();
				Color[] colors = UltiDraw.GetRainbowColors(2);

				for (int  j= 0; j < 2; j++) {
					if (contacts[j]!=Vector3.zero){
						UltiDraw.DrawSphere(contacts[j], Quaternion.identity, 0.1f, colors[j]);
					}
				}
				UltiDraw.End();
			}
		}
	}


	// private void AddChair(){
    //     // string [] files = System.IO.Directory.GetFiles("Assets/Resources/ShapeNet", "*.prefab");
    //     // Chair = GameObject.Find("Chair");
    //     // GameObject instance = Instantiate(Resources.Load("ShapeNet" + '\\' + files[Count / N].Split('\\').Last().Split('.')[0], typeof(GameObject))) as GameObject;
    //     instance.transform.SetParent(Chair.transform);
    //     // instance.transform.localScale += new Vector3(0.5f, 0.5f, 0.5f);
    //     instance.AddComponent<Interaction>();
    //     instance.GetComponent<VoxelCollider>().Generate();
    //     centre = instance.GetComponent<Interaction>().GetCenter().GetPosition();
    //     // Debug.Log(centre.ToString("F3"));

    // }

}
