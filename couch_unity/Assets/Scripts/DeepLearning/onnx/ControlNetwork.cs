using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;


public class ControlNetwork : MonoBehaviour
{
    public NNModel modelAsset;

    private Model m_RuntimeModel;
    public WorkerFactory.Device Device = WorkerFactory.Device.GPU;

    private IWorker worker;

    private bool verbose = false;

    public Tensor X = null;
    public Tensor Y = null;

    public Tensor h = null;
    public Tensor c = null;


    private float[] Y_float;

    int PropResolution = 8;
    private int Pivot = 0;


    void Start()
    {
        m_RuntimeModel = ModelLoader.Load(modelAsset, false, false);
        worker = m_RuntimeModel.CreateWorker(Device);
        // X = new Tensor(1, 1, 1, 122); /// no feet
        // X = new Tensor(1, 1, 1, 178); /// with feet 
        X = new Tensor(m_RuntimeModel.inputs.First().shape);

        h = new Tensor(2, 1, 256, 1);
        c = new Tensor(2, 1, 256, 1);

        for (int k = 0; k < h.length; k++)
        {
            h[k] = RandomFromDistribution.RandomNormalDistribution(0.0f, 0.01f);
        }
        for (int k = 0; k < c.length; k++)
        {
            c[k] = RandomFromDistribution.RandomNormalDistribution(0.0f, 0.01f);
        }

    }

   
    public void Predict() {
		// ContactSeries = (TimeSeries.Contact)TimeSeries.GetSeries("Contact");
        // Debug.Log(1);

        //Single Input
        var inputs = new Dictionary<string, Tensor>();

        inputs.Add("inputs", X);
        inputs.Add("h", h);
        inputs.Add("c", c);

        Y = worker.Execute(inputs).PeekOutput();

        // Y_float = new TimeSeries[Y.length];
        // for (int i=0; i<Y.length; i++){
        //     Y_float[i] = Y[i];
        // }
        

        // return hand_feature;
    }

    
		public void Feed(float value) {
            if(m_RuntimeModel != null) {
                if(Pivot == X.length) {
                    // Debug.Log(Session.Pivot);
                    // Debug.Log(value);

                    Debug.Log("Attempting to feed more values than inputs available.");
                } else {
                    X[Pivot] = value;
                    Pivot ++;
                }
            }
            // Debug.Log(Pivot);
		}

        public void Feed(bool value) {
            Feed(value ? 1f : 0f);
        }

        public void Feed(float[] values) {
            for(int i=0; i<values.Length; i++) {
                Feed(values[i]);
            }
        }

        public void Feed(bool[] values) {
            for(int i=0; i<values.Length; i++) {
                Feed(values[i]);
            }
        }

        public void Feed(Vector2 vector) {
            Feed(vector.x);
            Feed(vector.y);
        }

        public void Feed(Vector3 vector) {
            // Debug.Log(vector.x);
            Feed(vector.x);
            Feed(vector.y);
            Feed(vector.z);
        }

        public void FeedXY(Vector3 vector) {
            Feed(vector.x);
            Feed(vector.y);
        }

        public void FeedXZ(Vector3 vector) {
            Feed(vector.x);
            Feed(vector.z);
        }

        public void FeedYZ(Vector3 vector) {
            Feed(vector.y);
            Feed(vector.z);
        }


		public float Read() {
            float value = 0f;
            if(m_RuntimeModel != null) {
                // Debug.Log(Session.GetReadSize());

                if(Pivot == Y.length) {
                    Debug.Log("Attempting to read more values than outputs available.");
                } else {
                    value = Y[Pivot];
                    Pivot += 1;
                }
            }
            return value;
		}

    	public float Read(float min, float max) {
            return Mathf.Clamp(Read(), min, max);
		}

        public float[] Read(int count) {
            float[] values = new float[count];
            for(int i=0; i<count; i++) {
                values[i] = Read();
            }
            return values;
        }

        public float[] Read(int count, float min, float max) {
            float[] values = new float[count];
            for(int i=0; i<count; i++) {
                values[i] = Read(min, max);
            }
            return values;
        }

        public Vector3 ReadVector2() {
            return new Vector2(Read(), Read());
        }

        public Vector3 ReadVector3() {
            return new Vector3(Read(), Read(), Read());
        }

        public Vector3 ReadXY() {
            return new Vector3(Read(), Read(), 0f);
        }

        public Vector3 ReadXZ() {
            return new Vector3(Read(), 0f, Read());
        }

        public Vector3 ReadYZ() {
            return new Vector3(0f, Read(), Read());
        }

        public void SetPivot(int index) {
            Pivot = index;
        }

        public int GetPivot() {
            return Pivot;
        }

        public void ResetPivot() {
            // Debug.Log(Session.Pivot);
            Pivot = 0;
        }
    public void OnDestroy()
    {
        worker?.Dispose();
        Debug.Log("Destory being called");
        X.Dispose();
    }


}
            