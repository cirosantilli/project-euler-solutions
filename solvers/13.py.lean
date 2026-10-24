import «.formalsw».solvers.«13.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p13.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P13

namespace solvers.«13»
open FormalSW.PyAst
private abbrev program := Solvers.P13Ast.module

private abbrev integers (items : List Int) : List Mutable.Value :=
  items.map (fun n => .scalar (.int n))
private abbrev texts (items : List Int) : List Mutable.Value :=
  items.map (fun n => .scalar (.string (toString n)))
private abbrev stringBody (address : Nat) : List Stmt :=
  [.comprehensionAppend address (.builtinCall .str [.name "n"])]

private theorem string_loop (heap : Mutable.Heap) (address : Nat) (items remaining : List Int)
    (found : heap[address]? = some (.list (integers items))) (position : Nat)
    (suffix : items.drop position = remaining) (env : HeapCalls.Env) (done : List Int) :
    ∃ finalEnv, HeapCalls.BlockContinues program env
      [.sequenceStepLoop "n" (.ref address) position (stringBody heap.length) []]
      (heap ++ [.list (texts done)]) finalEnv (heap ++ [.list (texts (done ++ remaining))]) := by
  have inHeap : address < heap.length := List.getElem?_eq_some_iff.mp found |>.1
  have sequence : Mutable.sequence (.ref address) (heap ++ [.list (texts done)]) =
      some (.ok (integers items), heap ++ [.list (texts done)]) := by
    simp [Mutable.sequence, List.getElem?_append_left inHeap, found]
  induction remaining generalizing position env done with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.append_nil]
      apply HeapCalls.BlockContinues.sequence_done (source := .ref address) sequence
      · have absent : items[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [integers, absent]
      · exact HeapCalls.BlockContinues.nil
  | cons head tail ih =>
      let bound := HeapCalls.setLocal env "n" (.scalar (.int head))
      have emitted := HeapCalls.BlockContinues.comprehension_append (globals := [])
        (module := program) (env := bound) (address := heap.length)
        (heap := heap ++ [.list (texts done)])
        (HeapCalls.Evaluates.builtin (name := .str) (values := [.scalar (.int head)])
          (value := .scalar (.string (toString head)))
          (.cons (HeapCalls.Evaluates.lookup (name := "n") rfl) .nil) rfl)
        (by simp : (heap ++ [.list (texts done)])[heap.length]? = some (.list (texts done)))
      have step : HeapCalls.BlockContinues program bound (stringBody heap.length)
          (heap ++ [.list (texts done)]) bound (heap ++ [.list (texts (done ++ [head]))]) := by
        simpa [texts, List.map_append] using emitted
      obtain ⟨finalEnv, rest⟩ := ih (position + 1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) bound (done ++ [head])
        (by simp [Mutable.sequence, List.getElem?_append_left inHeap, found])
      refine ⟨finalEnv, ?_⟩
      have item : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      apply HeapCalls.BlockContinues.sequence_step (source := .ref address)
        (value := .scalar (.int head)) sequence (by simp [integers, item])
      simpa [stringBody, Calls.loopContinuation, Calls.loopControl, List.append_assoc] using step.append rest

private theorem string_list (heap : Mutable.Heap) (address : Nat) (items : List Int)
    (found : heap[address]? = some (.list (integers items))) (env : HeapCalls.Env)
    (lookup : env "nums" = some (.ref address)) :
    HeapCalls.Evaluates program env
      (.listComp (.builtinCall .str [.name "n"]) [("n", .name "nums", [])])
      heap (.ref heap.length) (heap ++ [.list (texts items)]) := by
  obtain ⟨finalEnv, executed⟩ := string_loop heap address items items found 0 rfl
    (HeapCalls.comprehensionEnv env [("n", .name "nums", [])]) []
  apply HeapCalls.Evaluates.list_comp_sequence (items := integers items)
    (by intros; simp) (HeapCalls.Evaluates.lookup lookup) (by simp [Mutable.sequence, found])
  simpa using executed

private theorem text_prefix (heap : Mutable.Heap) (text : String) (k : Nat) :
    Mutable.getSlice (.scalar (.string text)) (.scalar .none) (.scalar (.int k)) (.scalar .none) heap =
      some (.ok (.scalar (.string (String.ofList (text.toList.take k)))), heap) := by
  have slice := FormalSW.Py.Builtin.Sequence.slice_prefix text.toList (k : Int)
  rw [FormalSW.Py.Builtin.Slice.prefixLength_nonnegative, ← List.take_eq_take_min] at slice
  have converted : Strings.slice (.string text) .none (.int k) .none =
      .ok (.string (String.ofList (text.toList.take k))) := by
    change (match FormalSW.Py.Builtin.Sequence.slice text.toList { stop := some (k : Int) } with
      | .ok characters => Except.ok (Constant.string (String.ofList characters))
      | .error _ => Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)) = _
    rw [slice]
  change (match Strings.slice (.string text) .none (.int k) .none with
    | .ok value => some (Except.ok (Mutable.Value.scalar value), heap)
    | .error error => some (Except.error error, heap)) = _
  rw [converted]

/-- The original generator parses every printed input, sums exactly, and slices the total. -/
theorem first_digits.result (heap : Mutable.Heap) (address : Nat) (items : List Int) (k : Nat)
    (found : heap[address]? = some (.list (texts items))) :
    HeapCalls.ReturnsValue program "first_digits" [.ref address, .scalar (.int k)] heap
      (.scalar (.string (String.ofList ((toString (items.foldl Int.add 0)).toList.take k)))) heap := by
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.assign (value := .scalar (.int (items.foldl Int.add 0))) (middle := heap)
  · apply HeapCalls.Evaluates.sum_generator_sequence (items := texts items) (by intros; simp)
      (HeapCalls.Evaluates.lookup rfl) (by simp [Mutable.sequence, found]) HeapCalls.Evaluates.constant rfl
    apply HeapCalls.sum_map_loop (values := items) (encode := fun n => .scalar (.string (toString n)))
      (number := id) items (by decide)
    · simp [HeapCalls.sequenceValue, HeapCalls.sequenceSource, Mutable.sequence, found, texts]
    · intro env item lookup
      apply HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup lookup) .nil)
      change HeapCalls.scalarResult (Calls.integerConversion [.string (toString item)]) heap = _
      rw [Calls.integerConversion_int]
      rfl
    · rfl
    · simp [HeapCalls.setLocal]
  · apply HeapCalls.BlockReturns.return_value
    apply HeapCalls.Evaluates.slice
      (.cons (HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup rfl) .nil) rfl)
        (.cons HeapCalls.Evaluates.constant (.cons (HeapCalls.Evaluates.lookup rfl)
          (.cons HeapCalls.Evaluates.constant .nil))))
    exact text_prefix heap _ k


/-- The legacy ten-digit entry point delegates to the same generalized implementation. -/
theorem first_ten_digits.result (heap : Mutable.Heap) (address : Nat) (items : List Int)
    (found : heap[address]? = some (.list (texts items))) :
    HeapCalls.ReturnsValue program "first_ten_digits" [.ref address] heap
      (.scalar (.string (String.ofList ((toString (items.foldl Int.add 0)).toList.take 10)))) heap := by
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.return_value
  exact HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup rfl)
    (.cons HeapCalls.Evaluates.constant .nil)) (first_digits.result heap address items 10 found)

private theorem parsed_prefix (n k : Nat) (nonempty : (toString n).toList.take k ≠ []) :
    Calls.integerConversion [.string (String.ofList ((toString n).toList.take k))] =
      some (.ok (.int (ProjectEulerSolutions.P13.solve [n] k))) := by
  have digits : ∀ c ∈ (toString n).toList.take k, c.isDigit := by
    intro c member
    have original : c ∈ Nat.toDigits 10 n := by simpa using List.mem_of_mem_take member
    exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) original
  rw [Calls.integerConversion, FormalSW.Py.Builtin.Int.parseText_digits _ nonempty digits]
  simp [Calls.parsedInteger, ProjectEulerSolutions.P13.solve, ProjectEulerSolutions.P13.firstDigitsOfSum,
    ProjectEulerSolutions.P13.parseNat, Nat.ofDigitChars, Nat.mul_comm]

set_option maxHeartbeats 1000000 in
/-- The canonical integer-input adapter keeps the original string-based algorithm. -/
theorem solve.result (heap : Mutable.Heap) (address : Nat) (items : List Int) (n k : Nat)
    (found : heap[address]? = some (.list (integers items))) (summed : items.foldl Int.add 0 = (n : Int)) :
    HeapCalls.ReturnsValue program "solve" [.ref address, .scalar (.int k)] heap
      (.scalar (.int (ProjectEulerSolutions.P13.solve [n] k))) (heap ++ [.list (texts items)]) := by
  let env := HeapCalls.argumentEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    [.ref address, .scalar (.int k)]
  let middle := heap ++ [.list (texts items)]
  let leading := String.ofList ((toString n).toList.take k)
  have computed : HeapCalls.Evaluates program env
      (.call (.name "first_digits")
        [.listComp (.builtinCall .str [.name "n"]) [("n", .name "nums", [])], .name "k"])
      heap (.scalar (.string leading)) middle := by
    apply HeapCalls.Evaluates.call
      (.cons (string_list heap address items found env rfl) (.cons (HeapCalls.Evaluates.lookup rfl) .nil))
    simpa only [summed, show toString (n : Int) = toString n from rfl, leading] using
      first_digits.result middle heap.length items k (by simp [middle])
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.assign computed
  apply HeapCalls.BlockReturns.return_value
  by_cases empty : (toString n).toList.take k = []
  · apply HeapCalls.Evaluates.conditional (flag := false) (HeapCalls.Evaluates.lookup rfl)
    · simp only [leading, empty]
      rfl
    · have zero : ProjectEulerSolutions.P13.solve [n] k = 0 := by
        simp only [ProjectEulerSolutions.P13.solve, ProjectEulerSolutions.P13.firstDigitsOfSum,
          List.foldl_cons, List.foldl_nil, Nat.zero_add, ProjectEulerSolutions.P13.parseNat,
          String.toList_ofList, empty, List.foldl_nil]
      rw [zero]
      exact HeapCalls.Evaluates.constant
  · apply HeapCalls.Evaluates.conditional (flag := true) (HeapCalls.Evaluates.lookup rfl)
    · have nonemptyBool : leading.isEmpty = false := by
        simpa only [leading, String.isEmpty_eq_false_iff, ne_eq, String.ofList_eq_empty_iff] using empty
      change some (Except.ok (!leading.isEmpty), middle) = _
      rw [nonemptyBool]
      rfl
    · apply HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup rfl) .nil)
      change HeapCalls.scalarResult (Calls.integerConversion [.string leading]) middle = _
      rw [parsed_prefix n k empty]
      rfl

private abbrev numsEnv (address : Nat) : HeapCalls.Env :=
  fun name => if name = "nums" then some (.ref address) else none
private abbrev kEnv (k : Int) : HeapCalls.Env :=
  fun name => if name = "k" then some (.scalar (.int k)) else none

private theorem sum_condition (heap : Mutable.Heap) (address : Nat) (items : List Int)
    (found : heap[address]? = some (.list (integers items))) :
    HeapCalls.Evaluates program (numsEnv address) Solvers.P13Ast.requires_solve_0 heap
      (.scalar (.bool (decide (0 ≤ items.foldl Int.add 0)))) heap := by
  apply HeapCalls.Evaluates.compare (bv := .scalar (.int 0))
    (HeapCalls.Evaluates.sum_integers (values := items)
      (.cons (HeapCalls.Evaluates.lookup rfl) (.cons HeapCalls.Evaluates.constant .nil))
      (by simp [Mutable.sequence, found])) HeapCalls.Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

private theorem count_condition (heap : Mutable.Heap) (k : Int) :
    HeapCalls.Evaluates program (kEnv k) Solvers.P13Ast.requires_solve_1 heap
      (.scalar (.bool (decide (0 ≤ k)))) heap := by
  apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem requires_solve_0_supported (heap : Mutable.Heap) (address : Nat) (_k : Int)
    (_valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "list[int]") :
    (HeapCalls.evaluate program (numsEnv address) Solvers.P13Ast.requires_solve_0 heap).Supported := by
  obtain ⟨items, found⟩ := Mutable.Value.ints_of_hasType typed
  exact Execution.supported_of_finishes (sum_condition heap address items found).finishes

theorem requires_solve_1_supported (heap : Mutable.Heap) (address : Nat) (k : Int)
    (_valid : heap.Valid) (_typed : Mutable.Value.HasType heap (.ref address) "list[int]")
    (_sum : HeapCalls.Condition program (numsEnv address) Solvers.P13Ast.requires_solve_0 heap) :
    (HeapCalls.evaluate program (kEnv k) Solvers.P13Ast.requires_solve_1 heap).Supported :=
  Execution.supported_of_finishes (count_condition heap k).finishes

private theorem domain (heap : Mutable.Heap) (address : Nat) (k : Int)
    (typed : Mutable.Value.HasType heap (.ref address) "list[int]")
    (sum : HeapCalls.Condition program (numsEnv address) Solvers.P13Ast.requires_solve_0 heap)
    (count : HeapCalls.Condition program (kEnv k) Solvers.P13Ast.requires_solve_1 heap) :
    ∃ (items : List Int) (n size : Nat), heap[address]? = some (.list (integers items)) ∧
      items.foldl Int.add 0 = (n : Int) ∧ k = (size : Int) := by
  obtain ⟨items, found⟩ := Mutable.Value.ints_of_hasType typed
  have totalNonnegative : 0 ≤ items.foldl Int.add 0 := by
    simpa using (sum_condition heap address items found).condition_iff.mp sum
  have countNonnegative : 0 ≤ k := by simpa using (count_condition heap k).condition_iff.mp count
  exact ⟨items, (items.foldl Int.add 0).toNat, k.toNat, found,
    (Int.toNat_of_nonneg totalNonnegative).symm, (Int.toNat_of_nonneg countNonnegative).symm⟩

theorem solve.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (k : Int)
    (valid : heap.Valid) (typed : Mutable.Value.HasType heap (.ref address) "list[int]")
    (sum : HeapCalls.Condition program (numsEnv address) Solvers.P13Ast.requires_solve_0 heap)
    (count : HeapCalls.Condition program (kEnv k) Solvers.P13Ast.requires_solve_1 heap) :
    (HeapCalls.call program "solve" [.ref address, .scalar (.int k)] heap).Equiv
      (HeapCalls.call P13Ast.module "naive" [.ref address, .scalar (.int k)] heap) := by
  obtain ⟨items, n, size, found, summed, rfl⟩ := domain heap address k typed sum count
  have optimized := solve.result heap address items n size found summed
  obtain ⟨extra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p13.naive.sum_result
      heap address items n size found summed
  rw [ProjectEulerSolutions.P13.equiv [n] size] at reference
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap [.list (texts items)] extra
      (.scalar (.int (ProjectEulerSolutions.P13.solve [n] size))) [] valid trivial (by simp))

theorem solve.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat) (k : Int), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[int]" →
      HeapCalls.Condition program (numsEnv address) Solvers.P13Ast.requires_solve_0 heap →
      HeapCalls.Condition program (kEnv k) Solvers.P13Ast.requires_solve_1 heap →
      (HeapCalls.call program "solve" [.ref address, .scalar (.int k)] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat) (k : Int), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[int]" →
      HeapCalls.Condition program (numsEnv address) Solvers.P13Ast.requires_solve_0 heap →
      HeapCalls.Condition program (kEnv k) Solvers.P13Ast.requires_solve_1 heap →
      (HeapCalls.call P13Ast.module "naive" [.ref address, .scalar (.int k)] heap).Supported) := by
  constructor
  · intro heap address k _valid typed sum count
    obtain ⟨items, n, size, found, summed, rfl⟩ := domain heap address k typed sum count
    exact (solve.result heap address items n size found summed).returns.supported
  · intro heap address k _valid typed sum count
    obtain ⟨items, n, size, found, summed, rfl⟩ := domain heap address k typed sum count
    obtain ⟨extra, reference⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p13.naive.sum_result
        heap address items n size found summed
    exact reference.returns.supported

end solvers.«13»
