using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnController : MonoBehaviour {
    public GameObject[] anchors;
    public List<GameObject> wizards;
    int maxWizard = 10;
    int wizardCount = 0;
    static int wizardId = 0;
    bool stopSpawn = true;
    // Use this for initialization
    void Start () {
        InvokeRepeating("spawnWizard", 2, 2);
    }
    public void ToggleSpawning()
    {
        stopSpawn = !stopSpawn;
    }
    public void WizardDie()
    {
        wizardCount--;
    }
	void spawnWizard()
    {
     
        if (wizardCount > maxWizard || stopSpawn)
            return;
        int wizardIndex = Random.Range(0, wizards.Capacity);
        int index=Random.Range(0, 4);
        GameObject wizardObj = Instantiate(wizards[wizardIndex]);
        wizardObj.transform.position = anchors[index].transform.position;
        wizardObj.name = "wizard" + wizardId;
        wizardId++;
        wizardCount++;
      
    }
	// Update is called once per frame
	void Update () {
        
    }
}
