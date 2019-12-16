using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;
using Unity.Jobs;
using Unity.Burst;
using Random = Unity.Mathematics.Random;

public class Boids : MonoBehaviour {

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
        [ReadOnly]  public NativeArray<Vector3> position;
        [ReadOnly]  public NativeArray<Vector3> velocity;
        public NativeArray<Vector3> accel;
        public Vector3 weights;
        public float dstThreshold;
        int n => position.Length;

        public void Execute(int id) {
            Vector3 avgSpr = Vector3.zero,
                    avgVel = Vector3.zero,
                    avgPos = Vector3.zero,
                    pos    = position[id],
                    vel    = velocity[id];

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
            accel[id] += (avgSpr * weights.x + avgVel * weights.y + avgPos * weights.z) / n;
        }
    }

    [Unity.Burst.BurstCompile]
    struct CountJob : IJob {
        public NativeArray<Vector3> position;
        public NativeArray<int> result;
        int n => position.Length;

        public void Execute() {
            result[0] = 0;
            for (int i = 0; i < n; i++) {
                if(position[i].x > 0) result[0] += 1;
            }
        }
    }

    #endregion

    [SerializeField] protected int num;
    [SerializeField] protected GameObject prefab; 
    [SerializeField] protected Vector3 areaSize;
    [SerializeField] protected float distThreshold;
    [SerializeField] Vector2 velThreshold;
    [SerializeField] Vector3 simWeight;
    protected Transform[] objs;
    protected NativeArray<Vector3> pos, vel, acc;
    protected TransformAccessArray trs;
    protected NativeArray<Random>  rnd;
    protected NativeArray<int> rst;
    protected Random seed;

    void Start() {
        objs = new Transform[num];
        for (int i = 0; i < num; i++) {
            var obj = Instantiate(prefab).transform;
            obj.position = Vector3.zero;
            objs[i] = obj;
        }

        pos = new NativeArray<Vector3>(num, Allocator.Persistent);
        vel = new NativeArray<Vector3>(num, Allocator.Persistent);
        acc = new NativeArray<Vector3>(num, Allocator.Persistent);
        trs = new TransformAccessArray(objs);
        rst = new NativeArray<int>(1, Allocator.Persistent);
        rnd = new NativeArray<Random>(num, Allocator.Persistent);

        for (int i = 0; i < num; i++) {
            pos[i] = Vector3.zero;
            vel[i] = UnityEngine.Random.insideUnitSphere;
            acc[i] = Vector3.zero;
        }
        seed = new Random(1);
    }

    void Update() {
        for (int i = 0; i < num; i++) rnd[i] = new Random(seed.NextUInt());
        var jobWall = new UpdateWall { position = pos, accel = acc, scale = areaSize};
        var jobSmlt = new UpdateSmlt { position = pos, velocity = vel, accel = acc, dstThreshold = distThreshold, weights = simWeight};
        var jobMove = new UpdateMove { position = pos, velocity = vel, accel = acc, dt = Time.deltaTime, limit = velThreshold};
        var handlerWall = jobWall.Schedule(num, 0);
        var handlerSmlt = jobSmlt.Schedule(num, 0, handlerWall);
        var handlerMove = jobMove.Schedule(trs, handlerSmlt);
        handlerMove.Complete();
    }

    void OnDrawGizmos() {
        Gizmos.DrawWireCube(Vector3.zero, areaSize);
    }
    void OnGUI() {
    }

    void OnDestroy() {
        pos.Dispose();
        vel.Dispose();
        acc.Dispose();
        trs.Dispose();
        rst.Dispose();
        rnd.Dispose();
    }
}
