import solvers.«90.py.lean.d».Expansion
import solvers.«90.py.lean.d».Counting

/-! Complete optimized execution, with all temporary allocations separated from caller objects. -/
namespace solvers.«90».optimized
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions DigitDice
set_option Elab.async false
set_option maxHeartbeats 1200000

/-- Initialization, enumeration, expansion and pair counting preserve every caller object. -/
theorem body (faceCount address : Nat) (pairs : List can_display.Pair)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (pairs.map can_display.Pair.reference)))
    (valid : ∀ pair ∈ pairs, pair.Valid heap)
    (getCount : env "face_count" = some (number faceCount))
    (getPairs : env "required_pairs" = some (.ref address)) :
    ∃ extra, BlockReturns program env (masks.code.drop 1) heap
      (number (optimizedCount faceCount (pairs.map can_display.Pair.model))) (heap ++ extra) globals := by
  let initialEnv := setLocal env "masks" (.ref heap.length)
  obtain ⟨collectedEnv, garbage, collected, preserved⟩ := masks.enumeration faceCount heap [] [] initialEnv globals
    (by simpa [initialEnv, setLocal] using getCount) rfl
  let rawMasks := (choices faceCount (List.range 10)).map DigitMask.mask
  let collectedHeap := masks.working heap rawMasks garbage
  have masksFound : collectedHeap[heap.length]? = some (.list (rawMasks.map number)) :=
    HeapCollection.found heap _ garbage
  have getMasks : collectedEnv "masks" = some (.ref heap.length) := by
    rw [preserved "masks" (by decide) (by decide) (by decide)]
    rfl
  have keptPairs : collectedEnv "required_pairs" = some (.ref address) := by
    rw [preserved "required_pairs" (by decide) (by decide) (by decide)]
    simpa [initialEnv, setLocal] using getPairs
  let expandedMasks := rawMasks.map rotateMask
  let expandedHeap := collectedHeap ++ [.list (expandedMasks.map number)]
  let expandedEnv := setLocal collectedEnv "expanded" (.ref collectedHeap.length)
  let countedEnv := setLocal expandedEnv "count" (number 0)
  let readyEnv := setLocal countedEnv "n" (number rawMasks.length)
  let data : counting.Data := ⟨expandedMasks, pairs, collectedHeap.length, address⟩
  have retained {index : Nat} {object : Mutable.Object} (original : heap[index]? = some object) :
      expandedHeap[index]? = some object :=
    HeapCollection.prefix_found collectedHeap _ index object
      (HeapCollection.prefix_found heap _ index object original)
  have dataValid : counting.Valid data expandedHeap := by
    constructor
    · simp [data, expandedHeap]
    · exact retained found
    · intro pair member
      exact retained (valid pair member)
  have ready : counting.Ready data readyEnv := by
    constructor
    · simp [data, readyEnv, countedEnv, expandedEnv, setLocal]
    · simpa [data, readyEnv, countedEnv, expandedEnv, setLocal] using keptPairs
    · simp [data, readyEnv, expandedMasks, setLocal]
  have getRawMasks : countedEnv "masks" = some (.ref heap.length) := by
    simpa [countedEnv, expandedEnv, setLocal] using getMasks
  have rawKept := HeapCollection.prefix_found collectedHeap [.list (expandedMasks.map number)]
    heap.length _ masksFound
  have lengthValue : Evaluates program countedEnv (.builtinCall .len [.name "masks"])
      expandedHeap (number rawMasks.length) expandedHeap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getRawMasks) .nil)
    simp [primitive, Mutable.length, expandedHeap, rawKept, number]
  have counted := counting.result data expandedHeap globals dataValid 0 readyEnv ready rfl
  have result : BlockReturns program readyEnv [masks.code[6]'(by decide), masks.code[7]'(by decide)]
      expandedHeap (number (optimizedCount faceCount (pairs.map can_display.Pair.model))) expandedHeap globals := by
    simpa only [Nat.zero_add, data, counting.Data.test, expandedMasks, rawMasks, optimizedCount,
      List.map_map, Function.comp_def] using counted
  refine ⟨(.list (rawMasks.map number) :: garbage) ++ [.list (expandedMasks.map number)], ?_⟩
  have allocated : BlockContinues program env [masks.code[1]'(by decide)] heap initialEnv
      (heap ++ [.list []]) globals := BlockContinues.annAssign (Evaluates.list .nil)
  apply allocated.returns
  apply collected.returns
  have expanded : BlockContinues program collectedEnv [masks.code[3]'(by decide)] collectedHeap
      expandedEnv expandedHeap globals := BlockContinues.annAssign
        (expansion.result rawMasks heap.length collectedHeap collectedEnv globals masksFound getMasks)
  apply expanded.returns
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign lengthValue
  simpa only [expandedHeap, collectedHeap, masks.working, HeapCollection.working, List.append_assoc] using result

private theorem guard (faceCount : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (found : env "face_count" = some (.scalar (.int faceCount))) :
    Evaluates program env (.boolOp .or [
      .compare (.name "face_count") [(.lt, .constant (.int 0))],
      .compare (.name "face_count") [(.gt, .constant (.int 10))]]) heap
      (.scalar (.bool (decide (faceCount < 0 ∨ 10 < faceCount)))) heap globals := by
  have below : Evaluates program env (.compare (.name "face_count") [(.lt, .constant (.int 0))])
      heap (.scalar (.bool (decide (faceCount < 0)))) heap globals :=
    Evaluates.compare (Evaluates.lookup found) Evaluates.constant rfl
  have above : Evaluates program env (.compare (.name "face_count") [(.gt, .constant (.int 10))])
      heap (.scalar (.bool (decide (10 < faceCount)))) heap globals :=
    Evaluates.compare (Evaluates.lookup found) Evaluates.constant rfl
  by_cases negative : faceCount < 0
  · simpa only [negative, true_or, decide_true] using
      Evaluates.boolean_stop (op := BoolOp.or)
        (rest := [.compare (.name "face_count") [(.gt, .constant (.int 10))]])
        (by simpa only [negative, decide_true] using below) rfl rfl
  · apply Evaluates.boolean_next (flag := false)
      (by simpa only [negative, decide_false] using below) rfl rfl
    simpa only [negative, false_or] using Evaluates.boolean_last above rfl

/-- Valid decimal face counts return the optimized model, for arbitrarily long requirements. -/
theorem result (faceCount address : Nat) (bounded : faceCount ≤ 10) (pairs : List can_display.Pair)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list (pairs.map can_display.Pair.reference)))
    (valid : ∀ pair ∈ pairs, pair.Valid heap) :
    ∃ extra, ReturnsValue program "solve" [number faceCount, .ref address] heap
      (number (referenceCount faceCount (pairs.map can_display.Pair.model))) (heap ++ extra) globals := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let source := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := functionEnv args source [number faceCount, .ref address] globals
  have getCount : env "face_count" = some (number faceCount) := by cases globals <;> rfl
  have getPairs : env "required_pairs" = some (.ref address) := by cases globals <;> rfl
  obtain ⟨extra, executed⟩ := body faceCount address pairs heap globals env found valid getCount getPairs
  refine ⟨extra, ?_⟩
  rw [← optimizedCount_eq]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) (guard faceCount env heap globals getCount)
    (by have notBad : ¬ ((faceCount : Int) < 0 ∨ 10 < (faceCount : Int)) := by omega
        simp only [notBad, decide_false]; rfl)
  exact executed

/-- Impossible face counts return before allocating anything or reading the requirements. -/
theorem invalid (faceCount : Int) (impossible : faceCount < 0 ∨ 10 < faceCount)
    (requirements : Mutable.Value) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    ReturnsValue program "solve" [.scalar (.int faceCount), requirements] heap (number 0) heap globals := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := true)
    (guard faceCount _ heap globals (by cases globals <;> rfl))
    (by simp only [impossible, decide_true]; rfl)
  exact BlockReturns.return_value Evaluates.constant

/-- The complete optimized Python solver returns the signed mathematical answer for every face count. -/
theorem signed_result (faceCount : Int) (address : Nat) (pairs : List can_display.Pair)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list (pairs.map can_display.Pair.reference)))
    (valid : ∀ pair ∈ pairs, pair.Valid heap) :
    ∃ extra, ReturnsValue program "solve" [.scalar (.int faceCount), .ref address] heap
      (number (answer faceCount (pairs.map can_display.Pair.model))) (heap ++ extra) globals := by
  by_cases impossible : faceCount < 0 ∨ 10 < faceCount
  · refine ⟨[], ?_⟩
    simpa only [answer_invalid faceCount _ impossible, List.append_nil] using
      invalid faceCount impossible (.ref address) heap globals
  · have nonnegative : ¬ faceCount < 0 := by omega
    have cast : (faceCount.toNat : Int) = faceCount := by omega
    obtain ⟨extra, executed⟩ := result faceCount.toNat address (by omega) pairs heap globals found valid
    exact ⟨extra, by simpa only [answer, nonnegative, ↓reduceIte, number, cast] using executed⟩

end solvers.«90».optimized
