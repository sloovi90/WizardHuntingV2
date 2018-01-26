using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WizardBehaviour : MonoBehaviour {
    Vector3 dir;
    Quaternion toAngle;
    bool lerp = false;
    Vector3 reflect;
    GameObject enemy;
    Animator anim;
    bool seeEnemy=false;
    int life = 1;
    public enum WizardType { BLACK,WHITE,GREEN };
    public WizardType wizType;
    private void OnCollisionStay(Collision collision)
    {
        Debug.Log(name+" stay");
        transform.position += reflect * 0.1f;
    }
    bool IsMagicFatal(Collision collisionInfo)
    {
        return (collisionInfo.collider.gameObject.CompareTag("KillWizard1") && wizType == WizardType.BLACK) ||
            (collisionInfo.collider.gameObject.CompareTag("KillWizard2") && wizType == WizardType.WHITE) || 
            (collisionInfo.collider.gameObject.CompareTag("KillWizard3") && wizType == WizardType.GREEN);
    }
        void OnCollisionEnter(Collision collisionInfo)
    {
        Debug.Log(name);
        if(IsMagicFatal(collisionInfo))
        {
            Vector3 enemyToMe = (enemy.transform.position - transform.position);
            enemyToMe.y = 0;
            anim.SetTrigger("hit");
            anim.SetInteger("life", --life);
            GetComponent<Rigidbody>().velocity = enemyToMe.normalized * 120f * Time.deltaTime;
            toAngle = Quaternion.LookRotation(enemyToMe.normalized);
            return;
        }
        reflect = Vector3.Reflect(GetComponent<Rigidbody>().velocity, collisionInfo.contacts[0].normal).normalized;

        setMovement();
    }
    void setMovement()
    {
        float rotY = Random.Range(-30, 30);
        Vector3 v = (Quaternion.Euler(0, rotY, 0) * reflect).normalized;
        v.y = 0;
        v.Normalize();
        toAngle = Quaternion.LookRotation(v);
        if (v.x>0.1 || v.z>0.1)
            GetComponent<Rigidbody>().velocity = v * 100f * Time.deltaTime;
        else
        {
            GetComponent<Rigidbody>().velocity =  transform.forward * 100f * Time.deltaTime;
            toAngle = Quaternion.LookRotation(GetComponent<Rigidbody>().velocity);
        }

    }
    // Use this for initialization
    void Start () {
        foreach (GameObject o in GameObject.FindGameObjectsWithTag("mobs")) {
            if(o!=gameObject)
                Physics.IgnoreCollision(GetComponentInChildren<CapsuleCollider>(), o.GetComponentInChildren<CapsuleCollider>());
           }
        //setMovement();
        toAngle = transform.rotation;
        reflect = new Vector3(Random.value * 2 - 1, 0, Random.value * 2 - 1).normalized;
        while(reflect==Vector3.zero)
            reflect = new Vector3(Random.value * 2 - 1, 0, Random.value * 2 - 1).normalized;
        setMovement();
        anim = GetComponent<Animator>();
        anim.SetInteger("life", life);
        enemy = GameObject.Find("Camera (eye)");

    }

    // Update is called once per frame
    void Update()
    {
        if (life == 0)
        {
            GetComponentInChildren<CapsuleCollider>().enabled = false;
            GetComponent<Rigidbody>().velocity = Vector3.zero;
            DestroyObject(gameObject, 3f);
        }
        transform.rotation = Quaternion.Slerp(transform.rotation, toAngle, Time.deltaTime * 5f);
        Vector3 enemyToMe = (enemy.transform.position - transform.position);
        Vector3 v = GetComponent<Rigidbody>().velocity;
        if (v.x < 0.1 && v.z < 0.1)
        {

            GetComponent<Rigidbody>().velocity = transform.forward * 100f * Time.deltaTime;
            toAngle = Quaternion.LookRotation(GetComponent<Rigidbody>().velocity);
        }
        RaycastHit hit;
        if(Physics.Raycast(transform.position, enemyToMe,out hit,20f,9))
        {
            Debug.DrawRay(transform.position, enemyToMe, Color.red);
           // Debug.Log(enemy.transform.position);
            //Debug.Log("coliider " + hit.collider);
            enemyToMe.y = 0;
           
            if (hit.collider == enemy.GetComponent<BoxCollider>())
            {
                anim.SetBool("seeEnemy", true);
            }
           
            if (enemyToMe.magnitude < 2.5)
            {
                Debug.Log(name + "magnitude lower than 2.5");
                GetComponent<Rigidbody>().velocity = Vector3.zero;
                toAngle = Quaternion.LookRotation(enemyToMe.normalized);


            }
            if (enemyToMe.magnitude < 20 && enemyToMe.magnitude > 2.5)
            {
                Debug.Log(name + "magnitude higher than 2.5 and lower then 20");
                GetComponent<Rigidbody>().velocity = enemyToMe.normalized * 120f * Time.deltaTime;
                toAngle = Quaternion.LookRotation(enemyToMe.normalized);

            }
        }
        Vector3 distance = enemyToMe;
        distance.y = 0;
        anim.SetFloat("enemyDistance", distance.magnitude);

        RaycastHit hit2;
        if (Physics.Raycast(transform.position, -transform.up, out hit2)) {
            transform.position = hit2.point+new Vector3(0,0.3f,0);

        }
       
       
       
    }
    void OnDestroy()
    {
        GameObject.Find("GameController").GetComponent<SpawnController>().WizardDie();
    }
}
