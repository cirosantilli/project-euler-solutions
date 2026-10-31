import solvers.«83.py.lean.d».LoopControl
import ProjectEulerSolutions.Lib.DijkstraQueueSelection

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.TriangleBufferCalls (number)
open ProjectEulerSolutions.BinaryHeapOrder
set_option Elab.async false

theorem goal_branch (heap : Mutable.Heap) (env : HeapCalls.Env) (address index : Nat)
    (values : List Bool) (distance : Int)
    (found : heap[address]? = some (.list (encodeVisited values))) (inside : index < values.length)
    (waiting : values[index] = false)
    (localVisited : env "visited" = some (.ref address)) (localIndex : env "u" = some (number index))
    (localGoal : env "goal" = some (number index)) (localDistance : env "d" = some (number distance)) :
    BlockReturns program env ((loopBody.drop 1 ++ [loopResume]) ++ afterLoop) heap (number distance)
      (heap.set address (.list (encodeVisited (values.set index true)))) := by
  have read := visited_read heap env address index values found inside localVisited localIndex
  rw [waiting] at read
  change BlockReturns program env
    (.ifThen visitedExpressionAt [.continueLoop] [] :: (loopBody.drop 2 ++ [loopResume]) ++ afterLoop)
    heap (number distance) _
  apply BlockReturns.branch (flag := false) read rfl
  change BlockReturns program env
    ([.assign [visitedExpressionAt] (.constant (.bool true))] ++ (loopBody.drop 3 ++ [loopResume]) ++ afterLoop)
    heap (number distance) _
  apply (mark_visited heap env address index values found inside localVisited localIndex).returns
  have tested : Evaluates program env (.compare (.name "u") [(.eq, .name "goal")])
      (heap.set address (.list (encodeVisited (values.set index true)))) (.scalar (.bool true))
      (heap.set address (.list (encodeVisited (values.set index true)))) := by
    apply Evaluates.compare (Evaluates.lookup localIndex) (Evaluates.lookup localGoal)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  change BlockReturns program env
    (.ifThen (.compare (.name "u") [(.eq, .name "goal")]) [.return (.name "d")] [] ::
      (loopBody.drop 4 ++ [loopResume]) ++ afterLoop) _ (number distance) _
  apply BlockReturns.branch (flag := true) tested rfl
  exact BlockReturns.return_value (Evaluates.lookup localDistance)

/-- Compose the concrete pop, unpack, fresh-vertex test, visited write, and early return. -/
theorem loop_goal_return (heap : Mutable.Heap) (env : HeapCalls.Env) (queueAddress visitedAddress : Nat)
    (root : Mutable.Value) (rest : List Mutable.Value) (flags : List Bool)
    (found : heap[queueAddress]? = some (.list (root :: rest)))
    (visitedFound : heap[visitedAddress]? = some (.list (encodeVisited flags)))
    (separate : visitedAddress ≠ queueAddress)
    (typed : ∀ value ∈ root :: rest, QueueEntry heap value)
    (ordered : Ordered (priority heap) (root :: rest))
    (inside : vertex heap root < flags.length) (waiting : flags[vertex heap root] = false)
    (localHeap : env "heap" = some (.ref queueAddress))
    (localVisited : env "visited" = some (.ref visitedAddress))
    (localGoal : env "goal" = some (number (vertex heap root))) :
    ∃ after, BlockReturns program env loopTail heap (number (priority heap root)) after := by
  obtain ⟨middle, result, distance, index, key, indexEq, popped, resultFound, resultOrder, resultTyped, reordered⟩ :=
    pop_assignment heap env queueAddress root rest localHeap found typed ordered
  have middleFound : middle[visitedAddress]? = some (.list (encodeVisited flags)) := by
    rw [(Heapq.Reorders.frame heap queueAddress rest middle reordered).2 visitedAddress separate]
    exact visitedFound
  have indexInside : index < flags.length := by simpa only [indexEq] using inside
  have indexWaiting : flags[index] = false := by simpa only [indexEq] using waiting
  let ready := assignNames env ["d", "u"] [number distance, number index]
  have finished := goal_branch middle ready visitedAddress index flags distance middleFound indexInside indexWaiting
    (by simpa [ready, assignNames, setLocal] using localVisited) rfl
    (by simpa [ready, assignNames, setLocal, indexEq] using localGoal) rfl
  refine ⟨middle.set visitedAddress (.list (encodeVisited (flags.set index true))), ?_⟩
  rw [key]
  apply loop_enter heap _ env queueAddress root rest (number distance) localHeap found
  change BlockReturns program env ([popAssignment] ++ ((loopBody.drop 1 ++ [loopResume]) ++ afterLoop)) heap _ _
  exact popped.returns finished

/-- The concrete root selected by heap order attains a shortest walk when it is fresh. -/
theorem queue_root_minimum (matrix : List (List Int))
    (nonnegative : ∀ row ∈ matrix, ∀ value ∈ row, 0 ≤ value)
    (source : ProjectEulerSolutions.GridShortestPaths.Cell matrix)
    (state : ProjectEulerSolutions.P83.SearchState)
    (invariant : ProjectEulerSolutions.DijkstraGrid.Invariant matrix source state)
    (heap : Mutable.Heap) (root : Mutable.Value) (rest : List Mutable.Value)
    (covered : CoveredQueue heap (ProjectEulerSolutions.DijkstraGrid.total matrix)
      (ProjectEulerStatements.P83.bigVal matrix) (fun i => state.distances[i]!) (fun i => state.visited[i]!) (root :: rest))
    (ordered : Ordered (priority heap) (root :: rest)) (waiting : state.visited[vertex heap root]! = false) :
    ∃ selected : ProjectEulerSolutions.GridShortestPaths.Cell matrix,
      ProjectEulerSolutions.FiniteGridIndex.index selected = vertex heap root ∧
      ∃ shortest : (ProjectEulerSolutions.DijkstraGrid.grid matrix).Walk selected source,
        (shortest.support.map (ProjectEulerSolutions.GridShortestPaths.cost matrix)).sum = priority heap root ∧
        ∀ walk : (ProjectEulerSolutions.DijkstraGrid.grid matrix).Walk selected source,
          (shortest.support.map (ProjectEulerSolutions.GridShortestPaths.cost matrix)).sum ≤
            (walk.support.map (ProjectEulerSolutions.GridShortestPaths.cost matrix)).sum := by
  apply ProjectEulerSolutions.DijkstraGrid.queue_selected_minimum matrix nonnegative source state invariant
    (priority heap root, vertex heap root) (queueKeys heap rest) covered ?_ waiting
  intro entry member
  obtain ⟨value, present, rfl⟩ := (List.mem_map
    (f := fun value => (priority heap value, vertex heap value)) (l := root :: rest)).mp member
  exact root_le_member (priority heap) root rest ordered value present

end solvers.«83»
