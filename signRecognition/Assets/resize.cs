using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class resize : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
    void OnCollisionEnter(Collision collisionInfo)
    {
        Debug.Log(collisionInfo.collider.gameObject);
        if(!collisionInfo.collider.gameObject.CompareTag("mobs"))
                 Destroy(gameObject);
    }
    // Update is called once per frame
    void Update () {
        transform.localScale *= 1.003f;
        if ((GameObject.Find("Controller (left)").transform.position - transform.position).magnitude >100)
            DestroyImmediate(gameObject);
	}
}
