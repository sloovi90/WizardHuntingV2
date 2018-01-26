using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using UnityEditor;
using System.IO;
using System.Net.Sockets;
using System.Threading;
public class swipeTrail : MonoBehaviour {
	public SteamVR_TrackedController controller;
	public GameObject ctrl;
    public List<GameObject> magic;
	Plane objPlane;
	public string host="localhost";
	public int port=5656;
    TcpClient tcp=null;
    Vector3 actionPos;
    Vector3 targetPos;
    public GameObject aim;
    bool processAim = true;
    bool connected = false;
    //Color actionColor = Color.black;
    int actionIndex = -1;
    Texture2D RTImage(Camera cam,Vector3 min,Vector3 max) {
		//Debug.Log (min +""+max);
		float deltaX = 0.08f;
		float deltaY = 0.08f;
		Vector3 minC = cam.WorldToViewportPoint (min-new Vector3(deltaX,deltaY,0));
		Vector3 maxC = cam.WorldToViewportPoint(max+new Vector3(deltaX,deltaY,0));
		//Debug.Log (minC +""+maxC);
		RenderTexture currentRT = RenderTexture.active;
		RenderTexture.active = cam.targetTexture; 
		cam.Render();
		Texture2D image = new Texture2D (Mathf.CeilToInt(Mathf.Abs(maxC.x-minC.x)*Screen.width), Mathf.CeilToInt(Mathf.Abs(maxC.y-minC.y)*Screen.height));
		image.ReadPixels(new Rect(Screen.width/2-image.width/2,Screen.height/2-image.height/2, image.width,image.height), 0, 0);
		image.Apply();
		for (int i = 0; i < image.width; i++)
			for (int j = 0; j < image.height; j++)
				if (image.GetPixel (i, j) == Color.black)
					image.SetPixel (i, j, Color.black);
		GameObject.Find ("DemoCube").GetComponent<MeshRenderer> ().materials[0].mainTexture=image;
		//cam.targetTexture = null;
		RenderTexture.active = currentRT;
		return image;
	}


	public IEnumerator processImage(){
		TrailRenderer tr=GetComponent<TrailRenderer> ();
		LineRenderer ln =GameObject.Find("drawer").GetComponent<LineRenderer> ();
		Camera c1 = GameObject.Find ("Camera (eye)").GetComponent<Camera> ();
		Camera c2 = GameObject.Find ("SeconderyCamera").GetComponent<Camera> ();
		Debug.Log (c2);
		RenderTexture rt = new RenderTexture (Screen.width, Screen.height, 24);
		c2.targetTexture = rt;
		ln.positionCount = tr.positionCount ;
		Vector3[] arr=new Vector3[tr.positionCount];
		tr.GetPositions (arr);
		Vector3 center = ctrl.transform.position-tr.bounds.center;
		
		for(int i=0;i<tr.positionCount;i++)
			//move points to center of controller and then put them as local coordinates in front of camera
			arr[i] = ctrl.transform.InverseTransformPoint(arr[i]+center)+c2.gameObject.transform.position+c2.transform.forward;
		Vector3 min = arr [0], max = arr [0];
		//Debug.Log ("before Loop " + min + "" + max);
		foreach (Vector3 v in arr) {
			//Debug.Log (v);
			if (v.x < min.x)
				min.x = v.x;
			if (v.y < min.y)
				min.y = v.y;
			if (v.z < min.z)
				min.z = v.z;

			if (v.y > max.y)
				max.y = v.y;
			if (v.x > max.x)
				max.x = v.x;
			if (v.z > max.z)
				max.z = v.z;
		}
		

		ln.SetPositions (arr);

		LineRenderer ln2 = GameObject.Find ("Camera (eye)").GetComponent<LineRenderer> ();
		Vector3[] arr2 = { min, max };
		ln2.SetPositions (arr2);
		ln2.positionCount = 2 ;
        yield return new WaitForEndOfFrame ();
        Texture2D image =RTImage(c2,min,max);// new Texture2D(900, 500);

		File.WriteAllBytes (Application.dataPath+"\\..\\img.png", image.EncodeToPNG());
        NetworkStream stream = tcp.GetStream();
        StreamWriter writer = new StreamWriter(stream);

        writer.WriteLine('s');
        writer.Flush();
        new Thread(() =>
        {
            Debug.Log("Thread Started");
            StreamReader reader = new StreamReader(stream);
            string num = reader.ReadLine();
            Debug.Log(num);
            if (num == "5") actionIndex = 0;
            else if (num == "7") actionIndex = 1;
            else if (num == "8") actionIndex = 2;
            else processAim = true;
       
        }).Start();

        actionPos = tr.bounds.center; //ctrl.transform.position;// + 0.2f * dir;
 
        tr.Clear();
     //   ss();
		GetComponent<TrailRenderer> ().Clear ();
		GetComponent<TrailRenderer> ().enabled = false;


	}

	// Use this for initialization
	void draw(object sender,ClickedEventArgs e){
        processAim = false;
     
        GetComponent<TrailRenderer> ().enabled = true;
		
		
	}

	void processDrawing(object sender,ClickedEventArgs e){
        
		GetComponent<TrailRenderer> ().enabled = false;
		StartCoroutine ("processImage");
       

    }
    IEnumerator connect()
    {
        Debug.Log(Application.dataPath);
        System.Diagnostics.ProcessStartInfo info = new System.Diagnostics.ProcessStartInfo();
        info.CreateNoWindow = true;
        info.FileName = "python.exe";
        info.Arguments = "mnist\\mnist_user.py";
        info.UseShellExecute = false;
        System.Diagnostics.Process p = new System.Diagnostics.Process();
        p.StartInfo = info;
        p.Start();
        p.WaitForInputIdle();
        yield return new WaitForSeconds(2);
        tcp = new TcpClient(host, port);
        if (tcp.Connected)
        {
            connected = true;
        }
        
    }
	void Start () {
        StartCoroutine("connect");
        controller.TriggerClicked += draw;
		controller.TriggerUnclicked += processDrawing;
        
        
    }
	void aimControl()
    {
        RaycastHit hit;
        if (Physics.Raycast(transform.position, transform.forward, out hit) == true)
        {
            aim.transform.position = hit.point;
            targetPos = hit.point;
        }
        else
        {
            targetPos = ctrl.transform.position + 100 * ctrl.transform.forward;
            aim.transform.position = new Vector3(0, -20, 0);
        }
    }
	// Update is called once per frame
	void Update () {
        if (!connected)
            return;
        if (actionIndex !=-1)
        {
            GameObject ball = Instantiate(magic[actionIndex]);
            File.Delete(Application.dataPath + "\\..\\img.png");
            ball.transform.position = actionPos ;
            ball.GetComponent<Rigidbody>().AddForce((targetPos-actionPos).normalized * 1000, ForceMode.Acceleration);
            actionIndex = -1;
            actionPos = Vector3.zero;
            processAim = true;
        }
        if(processAim)
            aimControl();
       
    }

}
