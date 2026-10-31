import solvers.«83.py.lean.d».FreshIteration
import solvers.«83.py.lean.d».StaleIteration
import solvers.«83.py.lean.d».TerminalIteration
import solvers.«83.py.lean.d».LoopProgress

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions
open RectangularMatrixCalls (MatrixRows)
open TriangleBufferCalls (number)
open GridShortestPaths (Cell cost)
set_option Elab.async false

/-- The actual lazy-queue loop terminates at the goal and returns its shortest-path cost. -/
theorem loop_result (matrixAddress distanceAddress visitedAddress queueAddress : Nat)
    (rows : MatrixRows) (width : Nat) (source target : Cell (rows.map Prod.snd)) (answer : Int)
    (nonnegative : ∀ row ∈ rows.map Prod.snd, ∀ value ∈ row, 0 ≤ value)
    (reachable : ∀ v : Cell (rows.map Prod.snd), Nonempty ((DijkstraGrid.grid (rows.map Prod.snd)).Walk v source))
    (targetIndex : FiniteGridIndex.index target = rows.length * width - 1)
    (optimal : ∃ shortest : (DijkstraGrid.grid (rows.map Prod.snd)).Walk target source,
      (shortest.support.map (cost (rows.map Prod.snd))).sum = answer ∧
      ∀ walk : (DijkstraGrid.grid (rows.map Prod.snd)).Walk target source,
        answer ≤ (walk.support.map (cost (rows.map Prod.snd))).sum)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (values : List Int) (flags : List Bool) (queue : List Mutable.Value)
    (state : LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags queue)
    (locals : LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress rows.length width)
    (waiting : flags.getD (FiniteGridIndex.index target) false = false) :
    ∃ after, BlockReturns program env loopTail heap (number answer) after ∧ heap.length ≤ after.length ∧
      ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ visitedAddress → other ≠ queueAddress →
        after[other]? = heap[other]? := by
  suffices ∀ measure heap env values flags queue, loopMeasure flags queue = measure →
      LoopState heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags queue →
      LoopLocals env matrixAddress distanceAddress visitedAddress queueAddress rows.length width →
      flags.getD (FiniteGridIndex.index target) false = false →
      ∃ after, BlockReturns program env loopTail heap (number answer) after ∧ heap.length ≤ after.length ∧
        ∀ other, other < heap.length → other ≠ distanceAddress → other ≠ visitedAddress → other ≠ queueAddress →
          after[other]? = heap[other]? by
    exact this _ heap env values flags queue rfl state locals waiting
  intro measure
  induction measure using Nat.strong_induction_on with
  | h measure ih =>
      intro heap env values flags queue measured state locals waiting
      have nonempty := state.queue_nonempty heap matrixAddress distanceAddress visitedAddress queueAddress rows width
        source target values flags queue nonnegative reachable waiting
      cases queue with
      | nil => exact False.elim (nonempty rfl)
      | cons root rest =>
          have inside : vertex heap root < flags.length := by
            rw [state.flagsSize]
            exact state.root_inside heap matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags root rest
          by_cases done : flags.getD (vertex heap root) false = true
          · obtain ⟨middle, nextEnv, nextQueue, popped, nextState, nextLocals, indexLocal, count, size, popFrame⟩ :=
              stale_pop heap env matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags root rest
                state locals done
            have decrease : loopMeasure flags nextQueue < measure := by
              rw [← measured]
              exact stale_measure flags root rest nextQueue count
            obtain ⟨after, continued, lower, finalFrame⟩ :=
              ih _ decrease middle nextEnv values flags nextQueue rfl nextState nextLocals waiting
            have doneRead : flags[vertex heap root] = true := by
              rw [← List.getD_eq_getElem flags false inside]
              exact done
            have branch := stale_branch_of_next middle after nextEnv visitedAddress (vertex heap root) flags (number answer)
              nextState.visitedFound inside doneRead nextLocals.visited indexLocal continued
            refine ⟨after, ?_, by omega, ?_⟩
            · apply loop_enter heap after env queueAddress root rest _ locals.queue state.neighbors.queueFound
              exact popped.returns branch
            · intro other oldInside notDistance notVisited notQueue
              rw [finalFrame other (by omega) notDistance notVisited notQueue, popFrame other notQueue]
          · have fresh : flags.getD (vertex heap root) false = false := Bool.eq_false_iff.mpr done
            by_cases goal : vertex heap root = rows.length * width - 1
            · have value := state.goal_priority heap matrixAddress distanceAddress visitedAddress queueAddress rows width source target
                values flags root rest nonnegative fresh (targetIndex.trans goal.symm) answer optimal
              obtain ⟨after, returned, size, frame⟩ := goal_iteration heap env matrixAddress distanceAddress visitedAddress queueAddress
                rows width source values flags root rest state locals fresh goal
              rw [value] at returned
              exact ⟨after, returned, by omega, fun other _ _ notVisited notQueue => frame other notVisited notQueue⟩
            · obtain ⟨middle, nextEnv, nextValues, nextQueue, executed, nextState, nextLocals, count, lower, upper, stepFrame⟩ :=
                fresh_iteration heap env matrixAddress distanceAddress visitedAddress queueAddress rows width source values flags root rest
                  nonnegative state locals fresh goal
              have decrease : loopMeasure (flags.set (vertex heap root) true) nextQueue < measure := by
                rw [← measured]
                exact fresh_measure flags (vertex heap root) inside fresh root rest nextQueue count
              have different : vertex heap root ≠ FiniteGridIndex.index target := by rw [targetIndex]; exact goal
              have nextWaiting : (flags.set (vertex heap root) true).getD (FiniteGridIndex.index target) false = false := by
                simpa only [List.getD, List.getElem?_set_ne different] using waiting
              obtain ⟨after, continued, finalLower, finalFrame⟩ :=
                ih _ decrease middle nextEnv nextValues (flags.set (vertex heap root) true) nextQueue rfl nextState nextLocals nextWaiting
              have resumed : BlockReturns program nextEnv ([loopResume] ++ afterLoop) middle (number answer) after := by
                apply BlockReturns.loop_resume
                simpa only [← loop_source] using continued
              refine ⟨after, ?_, by omega, ?_⟩
              · apply loop_enter heap after env queueAddress root rest _ locals.queue state.neighbors.queueFound
                simpa only [List.append_assoc] using executed.returns resumed
              · intro other oldInside notDistance notVisited notQueue
                rw [finalFrame other (by omega) notDistance notVisited notQueue,
                  stepFrame other oldInside notDistance notVisited notQueue]

end solvers.«83»
