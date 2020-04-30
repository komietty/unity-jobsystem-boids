using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;
using Unity.Jobs;
using Unity.Burst;

public class BoidsNativeList : MonoBehaviour {

    #region Job
    [BurstCompile]
    struct UpdateMove : IJobParallelForTransform {
        public NativeArray<Vector3> position;
        public NativeArray<Vector3> velocity;
        public NativeArray<Vector3> accel;
        public Vector2 limit;
        public float dt;

        void IJobParallelForTransform.Execute(int id, TransformAccess t) {
            var vel = velocity[id] + accel[id] * dt;
            var dir = Vector3.Normalize(vel);
            var mag = Vector3.Magnitude(vel);
            vel = math.clamp(mag, limit.x, limit.y) * dir;
            t.position += vel * dt;
            t.rotation = Quaternion.LookRotation(dir);
            accel[id] = Vector3.zero;
            position[id] = t.position;
            velocity[id] = vel;
        }
    }

    [BurstCompile]
    struct UpdateWall : IJobParallelFor {
        [ReadOnly] public NativeArray<Vector3> position;
        public NativeArray<Vector3> accel;
        public Vector3 scale;

        public void Execute(int id) {
            var p = position[id];
            var s = scale * 0.5f;
            accel[id] += Condition(-s.x - p.x, Vector3.right) +
                         Condition(-s.y - p.y, Vector3.up) +
                         Condition(-s.z - p.z, Vector3.forward) +
                         Condition(+s.x - p.x, Vector3.left) +
                         Condition(+s.y - p.y, Vector3.down) +
                         Condition(+s.z - p.z, Vector3.back);
        }

        Vector3 Condition(float dst, Vector3 dir) {
            var threshold = 3f;
            var weight = 2f;
            var d = math.abs(dst);
            return d < threshold ? dir * (weight / (d / threshold)) : Vector3.zero;
        }
    }

    [BurstCompile]
    struct UpdateSmlt : IJobParallelFor {
        [ReadOnly] public NativeArray<Vector3> position;
        [ReadOnly] public NativeArray<Vector3> velocity;
        public NativeArray<Vector3> accel;
        public Vector3 weights;
        public float dstThreshold;
        int n => position.Length;

        public void Execute(int id) {
            Vector3 avgSpr = Vector3.zero,
                    avgVel = Vector3.zero,
                    avgPos = Vector3.zero,
                    pos = position[id],
                    vel = velocity[id];

            for (int j = 0; j < n; j++) {
                if (j == id) continue;
                var tgtPos = position[j];
                var tgtVel = velocity[j];
                var tgtDif = pos - tgtPos;
                if (tgtDif.magnitude < dstThreshold) {
                    avgSpr += tgtDif.normalized;
                    avgVel += tgtVel;
                    avgPos += tgtPos;
                }
            }

            accel[id] += avgSpr / n * weights.x + avgVel / n * weights.y + (avgPos / n - pos) * weights.z;
        }
    }

    [BurstCompile]
    struct UpdateBlck : IJob {
        public NativeArray<Vector3> position;
        public NativeArray<int> result;
        int n => position.Length;

        public void Execute() {
            for (int i = 0; i < 8; i++) result[i] = 0;
            for (int i = 0; i < n; i++) {
                var p = position[i];
                if (p.x < 0 && p.y < 0 && p.z < 0) result[0]++;
                else if (p.x > 0 && p.y < 0 && p.z < 0) result[1]++;
                else if (p.x < 0 && p.y > 0 && p.z < 0) result[2]++;
                else if (p.x < 0 && p.y < 0 && p.z > 0) result[3]++;
                else if (p.x > 0 && p.y > 0 && p.z < 0) result[4]++;
                else if (p.x > 0 && p.y < 0 && p.z > 0) result[5]++;
                else if (p.x < 0 && p.y > 0 && p.z > 0) result[6]++;
                else if (p.x > 0 && p.y > 0 && p.z > 0) result[7]++;
            }
        }
    }

    #endregion

    [SerializeField] protected int initNum;
    [SerializeField] protected GameObject prefab;
    [SerializeField] protected Vector3 areaSize;
    [SerializeField] protected float distThreshold;
    [SerializeField] protected Vector2 velThreshold;
    [SerializeField] protected Vector3 simWeight;
    [SerializeField] KeyCode add = KeyCode.A;
    [SerializeField] KeyCode remove = KeyCode.R;
    protected NativeList<Vector3> pos, vel, acc;
    protected TransformAccessArray trs;
    protected NativeArray<int> rst;
    protected int num;

    void Start() {
        pos = new NativeList<Vector3>(initNum, Allocator.Persistent);
        vel = new NativeList<Vector3>(initNum, Allocator.Persistent);
        acc = new NativeList<Vector3>(initNum, Allocator.Persistent);
        trs = new TransformAccessArray(initNum);
        rst = new NativeArray<int>(8, Allocator.Persistent);
        for (int i = 0; i < initNum; i++) AddInstance();
    }

    void AddInstance() {
        var obj = Instantiate(prefab).transform;
        obj.position = Vector3.zero;
        trs.Add(obj);
        pos.Add(Vector3.zero);
        vel.Add(UnityEngine.Random.insideUnitSphere);
        acc.Add(Vector3.zero);
        num++;
    }

    void Update() {
        // add instance
        if (Input.GetKey(add)) AddInstance();
        // remove instance
        if (num > 0 && Input.GetKey(remove)) {
            var i = UnityEngine.Random.Range(0, num);
            Destroy(trs[i].gameObject);
            pos.RemoveAtSwapBack(i);
            vel.RemoveAtSwapBack(i);
            acc.RemoveAtSwapBack(i);
            trs.RemoveAtSwapBack(i);
            num--;
        }

        var jobWall = new UpdateWall { position = pos.AsDeferredJobArray(), accel = acc.AsDeferredJobArray(), scale = areaSize };
        var jobSmlt = new UpdateSmlt { position = pos.AsDeferredJobArray(), velocity = vel.AsDeferredJobArray(), accel = acc.AsDeferredJobArray(), dstThreshold = distThreshold, weights = simWeight };
        var jobMove = new UpdateMove { position = pos.AsDeferredJobArray(), velocity = vel.AsDeferredJobArray(), accel = acc.AsDeferredJobArray(), dt = Time.deltaTime, limit = velThreshold };
        var jobBlck = new UpdateBlck { position = pos.AsDeferredJobArray(), result = rst };
        var handlerWall = jobWall.Schedule(num, 0);
        var handlerSmlt = jobSmlt.Schedule(num, 0, handlerWall);
        var handlerMove = jobMove.Schedule(trs, handlerSmlt);
        var handlerBlck = jobBlck.Schedule(handlerMove);
        handlerBlck.Complete();
    }

    void OnDrawGizmos() {
        Gizmos.DrawWireCube(Vector3.zero, areaSize);
    }

    string[] blocks = new string[] {
            "block-left-down-back",
            "block-right-down-back",
            "block-left-top-back",
            "block-left-down-foward",
            "block-right-top-back",
            "block-right-down-forward",
            "block-left-top-foward",
            "block-right-top-foward"
        };

    void OnGUI() {
        GUI.skin.label.fontSize = 20;
        GUI.Label(new Rect(10, 10, 300, 30), $"instance num: {num}");
        GUI.Label(new Rect(10, 30, 300, 30), $"----------------------------");
        for (int i = 0; i < 8; i++)
            GUI.Label(new Rect(10, i * 30 + 60, 300, 30), $"{blocks[i]}: {rst[i]}");
    }

    void OnDestroy() {
        pos.Dispose();
        vel.Dispose();
        acc.Dispose();
        trs.Dispose();
        rst.Dispose();
    }
}
