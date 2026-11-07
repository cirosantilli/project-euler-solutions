import solvers.«84.py.lean.d».StationarySign
import ProjectEulerSolutions.Lib.MonopolyIntegerWeights

namespace solvers.«84».stationary_weights
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

/-- Complete source execution returns integer stationary weights with a positive normalization scale. -/
theorem body_result (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (value : Mutable.Value) (view : transitions.View sides heap value)
    (getTransitions : env "transitions" = some value)
    (getDenominator : env "denominator" = some (.scalar (.int (256 * (sides : Int) * sides)))) :
    ∃ (values : Fin 120 → Int) (afterHeap : Mutable.Heap) (address : Nat),
      BlockReturns program env (executableBody definition.2) heap (.ref address) afterHeap globals ∧
      afterHeap[address]? = some (.list (List.ofFn fun index => .scalar (.int (values index)))) ∧
      Monopoly.IntegerWeights sides values ∧
      ∀ index, index < heap.length → afterHeap[index]? = heap[index]? := by
  obtain ⟨state, forwardEnv, forwardHeap, addresses, _, forward, forwardView, _, forwardMatrix, forwardSize, forwardObjects⟩ :=
    triangular_result sides positive heap env globals value view getTransitions getDenominator
  obtain ⟨ready, readyView⟩ := back_setup_result state forwardHeap heap.length addresses forwardView forwardEnv globals forwardMatrix forwardSize
  let readyEnv := backSetupEnv forwardEnv forwardHeap (state.matrix 119 119)
  have readyMatrix : readyEnv "matrix" = some (.ref heap.length) := by simpa [readyEnv, backSetupEnv, setLocal] using forwardMatrix
  have readySize : readyEnv "size" = some (.scalar (.int 120)) := by simpa [readyEnv, backSetupEnv, setLocal] using forwardSize
  have readyWeights : readyEnv "weights" = some (.ref (forwardHeap.length + 1)) := by simp [readyEnv, backSetupEnv, setLocal]
  have readyScale : readyEnv "scale" = some (.scalar (.int (state.matrix 119 119))) := by simp [readyEnv, backSetupEnv, setLocal]
  obtain ⟨backEnv, backHeap, backward, backView, locals, backObjects⟩ := back_loop_result state
    (forwardHeap ++ weightsObjects) heap.length (forwardHeap.length + 1) addresses readyView readyEnv globals
    readyMatrix readyWeights readySize readyScale
  obtain ⟨afterHeap, address, extra, returned, extension, found⟩ := sign_result state.matrix state.solution (state.matrix 119 119)
    backHeap heap.length (forwardHeap.length + 1) addresses backView backEnv globals
    ((locals _ (by decide)).trans readyWeights) ((locals _ (by decide)).trans readyScale)
  refine ⟨signedWeights (state.matrix 119 119) state.solution, afterHeap, address,
    forward.returns (ready.returns (backward.returns returned)), found, Monopoly.signed_solution sides state, ?_⟩
  intro index inside
  have initialFound := List.getElem?_eq_getElem inside
  have forwardFound := (forwardObjects index inside).trans initialFound
  have readyFound := HeapCollection.prefix_found forwardHeap weightsObjects index _ forwardFound
  have rootInside := (List.getElem?_eq_some_iff.mp forwardView.found).1
  have backFound := (backObjects index (by omega)).trans readyFound
  rw [extension]
  exact (HeapCollection.prefix_found backHeap extra index _ backFound).trans initialFound.symm

theorem result (sides : Nat) (positive : 0 < sides) (heap : Mutable.Heap)
    (value : Mutable.Value) (view : transitions.View sides heap value) (globals : Mutable.Globals) :
    ∃ (values : Fin 120 → Int) (afterHeap : Mutable.Heap) (address : Nat),
      ReturnsValue program "stationary_weights" [value, .scalar (.int (256 * (sides : Int) * sides))]
        heap (.ref address) afterHeap globals ∧
      afterHeap[address]? = some (.list (List.ofFn fun index => .scalar (.int (values index)))) ∧
      Monopoly.IntegerWeights sides values ∧
      ∀ index, index < heap.length → afterHeap[index]? = heap[index]? := by
  obtain ⟨values, afterHeap, address, executed, found, valid, preserved⟩ := body_result sides positive heap
    (functionEnv definition.1 definition.2 [value, .scalar (.int (256 * (sides : Int) * sides))] globals)
    globals value view (by cases globals <;> rfl) (by cases globals <;> rfl)
  exact ⟨values, afterHeap, address, BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl executed,
    found, valid, preserved⟩

end solvers.«84».stationary_weights
