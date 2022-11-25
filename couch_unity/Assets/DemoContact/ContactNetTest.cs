using System;
using System.IO;
using System.Threading;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Threading.Tasks;

public class ContactNetTest : MonoBehaviour
{
    public ContactNetwork ContactNet;
    GameObject Chair;
    int numOfContacts = 2;
    private int numOfObjects;

    public Vector3[][] contacts;
    public bool DrawContacts=true;

    private List<Vector3> candidates;
    private Vector3 centre;

    void Start()
    {
        Chair = GameObject.Find("Chair");
        numOfObjects = Chair.transform.childCount;
        

        contacts = new Vector3[numOfObjects][];
        for (int i = 0; i < numOfObjects; i++){
            contacts[i] = new Vector3[numOfContacts];
        }
        ContactNet.LoadDerived();


    }

    public void LateUpdate()
    {
        Utility.SetFPS(Mathf.RoundToInt(1));

        Inference();

    }

	public static bool InvalidClosestPointCheck(Collider collider) {
		if(collider.isTrigger) {
			//Debug.Log("Invalid Closest Point Check: " + collider.name + " Parent: " + collider.transform.parent.name + " Type: Trigger");
			return true;
		}
		if(collider is MeshCollider) {
			Debug.Log("Invalid Closest Point Check: " + collider.name + " Parent: " + collider.transform.name + " Parent: " + collider.transform.parent.name + " Type: MeshCollider");
			return true;
		}
		return false;
	}

	public  Vector3 GetClosestPointOverlapSphere(Vector3 center, float radius, LayerMask mask, out Collider collider) {
		Collider[] colliders = Physics.OverlapSphere(center, radius, mask);
		if(colliders.Length == 0) {
			collider = null;

			return center;
		}
		int pivot = 0;
		while(InvalidClosestPointCheck(colliders[pivot])) {
			pivot++;
			if(pivot == colliders.Length) {
				collider = null;

				return center;
			}
		}
		Vector3 point = colliders[pivot].ClosestPoint(center);
		float x = (point.x-center.x)*(point.x-center.x);
		float y = (point.y-center.y)*(point.y-center.y);
		float z = (point.z-center.z)*(point.z-center.z);
		float min = x + y + z;
		collider = colliders[pivot];
        candidates.Add(point);
        Debug.Log(min);

    
		for(int i=pivot+1; i<colliders.Length; i++) {
			if(!InvalidClosestPointCheck(colliders[pivot])) {
                // Debug.Log(i);
				Vector3 candidate = colliders[i].ClosestPoint(center);
                candidates.Add(candidate);

				x = (candidate.x-center.x)*(candidate.x-center.x);
				y = (candidate.y-center.y)*(candidate.y-center.y);
				z = (candidate.z-center.z)*(candidate.z-center.z);
				float d = x + y + z;
				if(d < min) {
					point = candidate;

					min = d;
					collider = colliders[i];
                    Debug.Log(min);

				}

			}
		}
        Debug.Log(min);
		return point;
	}

    private void Inference() {
        for (int i = 0; i < numOfObjects; i++) {   

            GameObject currGameObject = Chair.transform.GetChild(i).gameObject;
            Interaction interaction = currGameObject.GetComponent<Interaction>();

            if (interaction != null) {

                    Vector3[] tmp = ContactNet.PredictGoal(interaction, null);
                    
                    // Taking care of zeros
                    for (int j = 0; j < tmp.Length; j++) {   
                        if (tmp[j].y < 0.2f){
                            tmp[j] = Vector3.zero;
                            tmp[j].y = -1;

                        }
                    }

                    centre = currGameObject.GetComponent<Interaction>().GetCenter().GetPosition();

                    candidates = new List<Vector3>();

                    for (int j = 0; j < tmp.Length; j++) {   

                        BoxCollider[] voxels = currGameObject.GetComponentInChildren<VoxelCollider>().GetVoxels();
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
                        if (nearest.y < 0.2f){
                            nearest = Vector3.zero;
                            nearest.y = -1;

                        }
                        contacts[i][j] = nearest;

                    }


            }
            
        }
     }



    void OnRenderObject(){

		if (DrawContacts){
            UltiDraw.Begin();
            Color[] colors = UltiDraw.GetRainbowColors(numOfContacts);
            for (int i = 0; i < numOfObjects; i++){

                for (int  j= 0; j < numOfContacts; j++) {
                    if (contacts[i][j].y != -1f){
                        UltiDraw.DrawSphere(contacts[i][j], Quaternion.identity, 0.12f, colors[j]);
                        // Debug.Log(string.Format("{0} {1}", j, contacts[i][j].ToString("F3")));
                    }
                    
                }


            }

            UltiDraw.End();
        }
	}

 
   
}