import solvers.«18.py.lean.d».TriangleRuntime
import lang.py.FormalSW.HeapRuntimeCostSetup

namespace solvers.«18».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open HeapCalls.Runtime.NestedIntList (encode)
private abbrev program := Solvers.P18Ast.module

theorem rowWork_take_le (rows : TriangleRows) (n : Nat) : rowWork (rows.take n) ≤ rowWork rows := by
  induction rows generalizing n with
  | nil => simp [rowWork]
  | cons head tail ih =>
      cases n with
      | zero => simp [rowWork]
      | succ n =>
          have h := ih n
          simp only [List.take_succ_cons, rowWork, List.map_cons, List.sum_cons] at *
          omega

theorem rowWork_get_le (rows : TriangleRows) (i : Nat) (valid : i < rows.length) :
    rows[i].2.length ≤ rowWork rows := by
  induction rows generalizing i with
  | nil => simp at valid
  | cons head tail ih =>
      cases i with
      | zero => simp [rowWork]; omega
      | succ i =>
          have h := ih i (by simpa using valid)
          simp only [List.getElem_cons_succ, rowWork, List.map_cons, List.sum_cons] at *
          omega

private theorem prefix_read (heap extra : Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).choose]
  exact found

private def lastExpression : Expr :=
  .subscript (.name "triangle") (.unaryOp .uSub (.constant (.int 1)))

/-- The complete optimized source, including initialization and range bounds.
Input rows may share references; every allocated result stays in a fresh suffix. -/
theorem source_run (heap : Heap) (address : Nat) (rows : TriangleRows)
    (getTriangle : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : ∀ (i : Nat) (valid : i < rows.length), rows[i].2.length = i+1)
    (nonempty : rows ≠ []) :
    ∃ (extra : Heap) (answer : Int), ReturnsIn .word program "solve" [.ref address] heap
      (110*rowWork rows+299) (.scalar (.int answer)) (heap ++ extra) := by
  let args : List Arg := [⟨"triangle", some (.subscript (.name "list") (.subscript (.name "list") (.name "int"))), none⟩]
  let env := functionEnv args ((findFunctionIn program.body "solve").getD ([], [])).2 [.ref address] []
  have positive : 0 < rows.length := List.length_pos_iff.mpr nonempty
  have valid : rows.length-1 < rows.length := by omega
  let bottom := rows[rows.length-1]
  let extraBottom := [Mutable.Object.list (encode bottom.2)]
  let copiedHeap := heap ++ extraBottom
  let dpEnv := setLocal env "dp" (.ref heap.length)
  have shifted : (-1 + (rows.length : Int)).toNat = rows.length-1 := by omega
  have indexed : Mutable.index (rows.map (fun pair => Mutable.Value.ref pair.1)) (.scalar (.int (-1))) heap =
      some (.ok (rows.length-1), heap) := by
    have inBounds : ¬ (-1 + (rows.length : Int) < 0 ∨ (rows.length : Int) ≤ -1 + (rows.length : Int)) := by omega
    simp [Mutable.index, shifted, inBounds]
  have last : EvaluatesIn .word program env lastExpression heap 13 (.ref bottom.1) heap := by
    apply EvaluatesIn.list_subscript (ca := 1) (cb := 4)
      (by intros a b c h; cases h) (EvaluatesIn.lookup rfl)
      (EvaluatesIn.negate_int EvaluatesIn.constant) getTriangle
    simp [Mutable.getItem, getTriangle, indexed, Except.map, List.getElem?_eq_getElem valid, bottom]
  have copied : EvaluatesIn .word program env (.subscript lastExpression (.slice none none none)) heap
      (7*bottom.2.length+229) (.ref heap.length) copiedHeap := by
    have run := EvaluatesIn.list_slice (lower := none) (upper := none) (stride := none)
      (ArgumentsIn.cons last (ArgumentsIn.cons EvaluatesIn.constant
        (ArgumentsIn.cons EvaluatesIn.constant (ArgumentsIn.cons EvaluatesIn.constant ArgumentsIn.nil))))
      (getRows bottom (List.getElem_mem valid))
      (Mutable.getSlice_list_copy heap bottom.1 (encode bottom.2) (getRows bottom (List.getElem_mem valid)))
      (SequenceWork.slice_full (encode bottom.2).length)
    exact run.mono (by simp; omega)
  obtain ⟨extra, answer, loop⟩ := triangle_loop (rows.length-1) copiedHeap dpEnv address heap.length rows bottom.2
    (prefix_read heap extraBottom _ _ getTriangle)
    (fun pair member => prefix_read heap extraBottom _ _ (getRows pair member)) shape (by omega)
    (by simp [copiedHeap, extraBottom]) (shape _ valid) rfl rfl
  have length : EvaluatesIn .word program dpEnv (.builtinCall .len [.name "triangle"]) copiedHeap 5
      (.scalar (.int rows.length)) copiedHeap := by
    apply EvaluatesIn.builtin_single (builtin := .len) (work := 1)
      (by intro h; cases h) (by intro h; cases h) (EvaluatesIn.lookup rfl)
    · simp [HeapCalls.primitive, Mutable.length, prefix_read heap extraBottom _ _ getTriangle, copiedHeap]
    · simp [operation, sequenceLength, prefix_read heap extraBottom _ _ getTriangle, copiedHeap]
  have start : EvaluatesIn .word program dpEnv
      (.binOp (.builtinCall .len [.name "triangle"]) .sub (.constant (.int 2))) copiedHeap 10
      (.scalar (.int ((rows.length : Int)-2))) copiedHeap :=
    EvaluatesIn.binary (op := .sub) (work := 1) length EvaluatesIn.constant rfl rfl
  have bounds := BlockReturnsIn.for_range_triple start
    (EvaluatesIn.negate_int (EvaluatesIn.constant (value := .int 1)))
    (EvaluatesIn.negate_int (EvaluatesIn.constant (value := .int 1))) (by decide)
    (by simpa only [show (rows.length : Int)-2 = ((rows.length-1 : Nat) : Int)-1 by omega] using loop)
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := args) (values := [.ref address]) rfl rfl (BlockReturnsIn.assign copied bounds)
  refine ⟨extraBottom ++ extra, answer, fun anchors stack => ?_⟩
  have tailWork := rowWork_take_le rows (rows.length-1)
  have bottomWork := rowWork_get_le rows (rows.length-1) valid
  have budget : (1+1+1)+(7*bottom.2.length+229)+(10+4+4+(100*rowWork (rows.take (rows.length-1))+15)+11)+2 ≤
      110*rowWork rows+299 := by change bottom.2.length ≤ rowWork rows at bottomWork; omega
  have bounded := (called anchors stack).mono (large := 110*rowWork rows+299)
    (by simp only [args, List.length_cons, List.length_nil, PrimitiveWork.integer]; omega)
  simpa only [copiedHeap, List.append_assoc] using bounded

theorem rowWork_bindings (rows : List (List Int)) :
    rowWork (NestedIntList.bindings rows) = NestedIntList.size rows := by
  have total : ∀ pairs : TriangleRows, rowWork pairs =
      (pairs.map Prod.snd).length + ((pairs.map Prod.snd).map List.length).sum := by
    intro pairs
    induction pairs with
    | nil => rfl
    | cons head tail ih => simp only [rowWork, List.map_cons, List.sum_cons, List.length_cons] at *; omega
  simpa only [NestedIntList.bindings_values, NestedIntList.size] using total (NestedIntList.bindings rows)

theorem source_bound (heap : Heap) (address : Nat) (rows : TriangleRows)
    (getTriangle : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : ∀ (i : Nat) (valid : i < rows.length), rows[i].2.length = i+1)
    (nonempty : rows ≠ []) :
    TimeBound .word program "solve" [.ref address] heap (110*rowWork rows+300) := by
  obtain ⟨extra, answer, run⟩ := source_run heap address rows getTriangle getRows shape nonempty
  exact run.timeBound_append

end solvers.«18».runtimeCost
