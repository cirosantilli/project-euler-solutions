import solvers.«83.py.lean.d».QueueInvariant
import solvers.«83.py.lean.d».InitialState

namespace solvers.«83»
open FormalSW.PyAst
set_option Elab.async false

theorem initial_distance_read (size index : Nat) (bound distance : Int) (positive : 0 < size) :
    (initialDistances size bound distance).getD index bound = if index = 0 then distance else bound := by
  by_cases zero : index = 0
  · subst index
    simp [initialDistances, List.getD, positive]
  · by_cases inside : index < size <;>
      simp [initialDistances, List.getD, zero, Ne.symm zero, inside]

theorem initial_covered (heap : Mutable.Heap) (size : Nat) (bound distance : Int)
    (positive : 0 < size) (finite : distance < bound) :
    CoveredQueue (initialHeap heap size bound distance) size bound
      (fun index => (initialDistances size bound distance).getD index bound)
      (fun index => (List.replicate size false).getD index false) [.ref (heap.length + 2)] := by
  have values : (fun index => (initialDistances size bound distance).getD index bound) =
      fun index => if index = 0 then distance else bound :=
    funext fun index => initial_distance_read size index bound distance positive
  have flags : (fun index => (List.replicate size false).getD index false) = fun _ => false := by
    funext index
    by_cases inside : index < size <;> simp [List.getD, inside]
  have tupleFound : (initialHeap heap size bound distance)[heap.length + 2]? =
      some (.tuple [.scalar (.int distance), .scalar (.int (0 : Nat))]) := by simp [initial_heap_layout]
  have keys : queueKeys (initialHeap heap size bound distance) [.ref (heap.length + 2)] = [(distance, 0)] := by
    simp only [queueKeys, List.map_cons, List.map_nil, priority_read _ _ distance 0 tupleFound,
      vertex_read _ _ distance 0 tupleFound]
  unfold CoveredQueue
  rw [values, flags, keys]
  exact ProjectEulerSolutions.LazyDijkstraQueue.initial (fun v => v < size) 0 bound distance positive finite

end solvers.«83»
