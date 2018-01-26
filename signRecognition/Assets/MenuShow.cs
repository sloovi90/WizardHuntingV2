using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MenuShow : MonoBehaviour {
    SteamVR_TrackedController controller;
	// Use this for initialization
	void Start () {
        controller = GetComponent<SteamVR_TrackedController>();
        controller.MenuButtonClicked += ToggleMenuShow;
	}
	void ToggleMenuShow(object s,ClickedEventArgs e)
    {
        GameObject.Find("GameController").GetComponent<Menu>().ToggleMenuOnOff();
       
    }
	// Update is called once per frame
	void Update () {
		
	}
}
