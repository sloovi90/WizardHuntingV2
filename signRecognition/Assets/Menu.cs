using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Menu : MonoBehaviour {
    public GameObject menu;
    List<GameObject> mobs;
	// Use this for initialization
	void Start () {
        mobs = new List<GameObject>();
	}
    public void ToggleMenuOnOff()
    {
        GetComponent<SpawnController>().ToggleSpawning();
        menu.SetActive(!menu.activeSelf);
        if (mobs.Count != 0)
        {
            foreach (GameObject o in mobs)
            {

                o.SetActive(!menu.activeSelf);
            }
            mobs.Clear();
        }
        else
        {
            foreach (GameObject o in GameObject.FindGameObjectsWithTag("mobs"))
            {
                mobs.Add(o);
                o.SetActive(!menu.activeSelf);
            }
        }
    }
        
    
	// Update is called once per frame
	void Update () {
		
	}
}
