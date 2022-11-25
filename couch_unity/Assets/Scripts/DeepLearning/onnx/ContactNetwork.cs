using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;


public class ContactNetwork : MonoBehaviour
{
    public NNModel modelAsset;

    private Model m_RuntimeModel;
    private IWorker worker;

    private bool verbose = false;

    int Xsize = 2064;
    private int Ysize = 6;
    private int Zsize = 16;



    private Tensor Y = null;

    Tensor X, z;
    Tensor Xmean, Xstd, Ymean, Ystd;

    int PropResolution = 8;

    
    private int Pivot = 0;

    public void LoadDerived()
    {
        if (modelAsset != null)
        {
            X = new Tensor(1, Xsize);
            m_RuntimeModel = ModelLoader.Load(modelAsset, verbose);
            worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, m_RuntimeModel, verbose);


            // z = new Tensor(1, Zsize);
// 
        }

    }

    public CuboidMap GetInteractionGeometry(Interaction interaction)
    {
        CuboidMap sensor = new CuboidMap(new Vector3Int(PropResolution, PropResolution, PropResolution));
        sensor.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents());
        return sensor;
    }

    public void Predict() {
		// ContactSeries = (TimeSeries.Contact)TimeSeries.GetSeries("Contact");
        // Debug.Log(1);

        //Single Input
        var inputs = new Dictionary<string, Tensor>();

        inputs.Add("x", X);
        // inputs.Add("z", z);

        Y = worker.Execute(inputs).PeekOutput();

        // Y_float = new TimeSeries[Y.length];
        // for (int i=0; i<Y.length; i++){
        //     Y_float[i] = Y[i];
        // }
        

        // return hand_feature;
    }


    // public CuboidMap GetInteractionGeometry(Interaction interaction)
    // {
    //     CuboidMap sensor = new CuboidMap(new Vector3Int(PropResolution, PropResolution, PropResolution));
    //     sensor.Sense(interaction.GetCenter(), LayerMask.GetMask("Interaction"), interaction.GetExtents());
    //     return sensor;
    // }

    public Vector3[] PredictGoal(Interaction interaction, string name)
    {   
        ResetPivot();
        // if (name != null)
        // {
        //     if (interaction.GetContactTransform("Hips_Pred") != null)
        //         interaction.RemoveContact();
        // }

        Matrix4x4 root = interaction.GetCenter();

        //Prepare input
        CuboidMap interactionGeometry = GetInteractionGeometry(interaction);



        for (int k = 0; k < Zsize; k++) {
            ///////////////////////////////////////////////////// This was originally 2!!!!!!!!!!!!!!!
            this.Feed(RandomFromDistribution.RandomNormalDistribution(0.0f, 1.0f) * 2);
        }

        for (int k = 0; k < interactionGeometry.Points.Length; k++) {
            this.Feed(interactionGeometry.References[k].GetRelativePositionTo(root));
            this.Feed(interactionGeometry.Occupancies[k]);
        }

        var inputs = new Dictionary<string, Tensor>();
        inputs.Add("x", X);
        //Run model
        worker.Execute(inputs);
        // Get output
        Tensor output = worker.PeekOutput();


        //Parse output
        var rh_pos = new Vector3(output[0], output[1], output[2]);
        rh_pos = rh_pos.GetRelativePositionFrom(root);
        var lh_pos = new Vector3(output[3], output[4], output[5]);
        lh_pos = lh_pos.GetRelativePositionFrom(root);
        // var hip_rot = Quaternion.LookRotation(Vector3.ProjectOnPlane(hip_forward, Vector3.up).normalized, Vector3.up);

        // if (name != null) { interaction.AddContact("Hips_Pred", hip_pos, hip_rot); }
        // else { interaction.AddContact(hip_pos, hip_rot); }
        
        Vector3[] contacts = new Vector3[2];
        contacts[0] = rh_pos;
        contacts[1] = lh_pos;
        return contacts;
    }
    
    public void Feed(float value) {
        if(m_RuntimeModel != null) {
            if(Pivot == X.length) {
                Debug.Log(Pivot);

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
            