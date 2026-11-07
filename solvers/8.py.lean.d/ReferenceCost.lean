import data.«project-euler-statements».data.py.project_euler_statements.«p8.py»
import lang.py.FormalSW.AllocationCostCalls

/-! Weighted source execution for Euler 8's reference on arbitrary integer lists.
The cost includes all list copies and products, uniformly over window widths. -/
namespace solvers.«8».referenceCost
open FormalSW.PyAst
open data.«project-euler-statements».data.py.project_euler_statements.p8
open HeapCalls.AllocationCost
open HeapCalls.AllocationCost.Calls
private abbrev program := P8Ast.module
private def ints (xs : List Int) : List Mutable.Value := xs.map fun x => .scalar (.int x)
private abbrev productBody : List Stmt := [.augAssign (.name "p") .mult (.name "x")]

private theorem product.loop (remaining : List Int) (before : List Int) (heap : Mutable.Heap)
    (address : Nat) (found : heap[address]? = some (.list (ints (before ++ remaining))))
    (env : HeapCalls.Env) (acc : Int) (hp : env "p" = some (.scalar (.int acc))) :
    BlockReturnsIn program env
      [.sequenceStepLoop "x" (.ref address) before.length productBody [], .return (.name "p")]
      heap (7 * remaining.length + 3) (.scalar (.int (acc * remaining.prod))) heap := by
  induction remaining generalizing before env acc with
  | nil =>
      simp only [List.length_nil, Nat.mul_zero, Nat.zero_add]
      apply BlockReturnsIn.sequence_done (items := ints before)
      · simp only [List.append_nil] at found
        simp [HeapCalls.sequenceValue, Mutable.sequence, found]
      · simp [ints]
      · simpa [List.prod_nil] using BlockReturnsIn.return_value (EvaluatesIn.lookup hp)
  | cons value rest ih =>
      simp only [List.length_cons, Nat.mul_add, Nat.mul_one]
      apply BlockReturnsIn.sequence_step (items := ints (before ++ value :: rest)) (value := .scalar (.int value))
      · simp [HeapCalls.sequenceValue, Mutable.sequence, found]
      · simp [ints]
      · rw [show 7 * rest.length + 9 = 1 + (7 * rest.length + 3) + 5 by omega]
        apply BlockReturnsIn.augAssign (cb := 7 * rest.length + 3) (by simpa [HeapCalls.setLocal] using hp)
          (EvaluatesIn.lookup rfl) (middle := heap) (right := .scalar (.int value))
          (value := .scalar (.int (acc * value)))
        · rfl
        · have next := ih (before ++ [value]) (by simpa [List.append_assoc] using found)
            (HeapCalls.setLocal (HeapCalls.setLocal env "x" (.scalar (.int value)))
              "p" (.scalar (.int (acc * value)))) (acc * value) rfl
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, productBody, FormalSW.PyAst.Calls.loopContinuation, FormalSW.PyAst.Calls.loopControl, FormalSW.PyAst.Calls.loopControlStmt,
            List.length_append, List.prod_cons, Int.mul_assoc] using next

theorem list_product.result (xs : List Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ints xs))) :
    ReturnsIn program "list_product" [.ref address] heap (7 * xs.length + 10) (.scalar (.int (xs.prod))) heap := by
  apply BlockReturnsIn.function rfl rfl
  rw [show 7 * xs.length + 9 = 1 + (7 * xs.length + 6) + 2 by omega]
  apply BlockReturnsIn.assign (cb := 7 * xs.length + 6) EvaluatesIn.constant
  apply BlockReturnsIn.for_sequence_name rfl (items := ints xs)
  · simp [Mutable.sequence, found]
  · let env := HeapCalls.setLocal
      (HeapCalls.argumentEnv [{ name := "xs", annotation := some (.subscript (.name "list") (.name "int")) }]
        [.ref address]) "p" (.scalar (.int 1))
    simpa using product.loop xs [] heap address (by simpa using found) env 1 rfl

private abbrev maximumBody : List Stmt :=
  [.assign [.name "m"] (.builtinCall .max [.name "m", .name "x"])]

private theorem maximum_operation (a b : Int) (heap : Mutable.Heap) :
    HeapCalls.primitive (.builtin .max) [.scalar (.int a), .scalar (.int b)] heap =
      some (.ok (.scalar (.int (max a b))), heap) := by
  simp [HeapCalls.primitive, HeapCalls.scalarResult, Calls.positionalMaximum,
    Calls.maximumFrom, scalarCompare, evalCmpOp, max_def]
  split <;> split <;> simp_all <;> omega

private theorem maximum.loop (remaining before : List Int) (heap : Mutable.Heap)
    (address : Nat) (found : heap[address]? = some (.list (ints (before ++ remaining))))
    (env : HeapCalls.Env) (acc : Int) (hm : env "m" = some (.scalar (.int acc))) :
    BlockReturnsIn program env
      [.sequenceStepLoop "x" (.ref address) before.length maximumBody [], .return (.name "m")]
      heap (9 * remaining.length + 3) (.scalar (.int (remaining.foldl max acc))) heap := by
  induction remaining generalizing before env acc with
  | nil =>
      simp only [List.length_nil, Nat.mul_zero, Nat.zero_add]
      apply BlockReturnsIn.sequence_done (items := ints before)
      · simp only [List.append_nil] at found
        simp [HeapCalls.sequenceValue, Mutable.sequence, found]
      · simp [ints]
      · exact BlockReturnsIn.return_value (EvaluatesIn.lookup hm)
  | cons value rest ih =>
      simp only [List.length_cons, Nat.mul_add, Nat.mul_one]
      apply BlockReturnsIn.sequence_step
        (items := ints (before ++ value :: rest)) (value := .scalar (.int value))
      · simp [HeapCalls.sequenceValue, Mutable.sequence, found]
      · simp [ints]
      · rw [show 9 * rest.length + 11 = 6 + (9 * rest.length + 3) + 2 by omega]
        apply BlockReturnsIn.assign (ca := 6) (cb := 9 * rest.length + 3)
        · exact EvaluatesIn.builtin
            (.cons (EvaluatesIn.lookup (by simpa [HeapCalls.setLocal] using hm))
              (.cons (EvaluatesIn.lookup rfl) .nil)) (maximum_operation acc value heap)
        · have next := ih (before ++ [value]) (by simpa [List.append_assoc] using found)
            (HeapCalls.setLocal (HeapCalls.setLocal env "x" (.scalar (.int value)))
              "m" (.scalar (.int (max acc value)))) (max acc value) rfl
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, maximumBody, FormalSW.PyAst.Calls.loopContinuation, FormalSW.PyAst.Calls.loopControl, FormalSW.PyAst.Calls.loopControlStmt,
            List.length_append] using next

theorem list_max.result (xs : List Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ints xs))) :
    ReturnsIn program "list_max" [.ref address] heap (9 * xs.length + 10)
      (.scalar (.int (xs.foldl max 0))) heap := by
  apply BlockReturnsIn.function rfl rfl
  rw [show 9 * xs.length + 9 = 1 + (9 * xs.length + 6) + 2 by omega]
  apply BlockReturnsIn.assign (cb := 9 * xs.length + 6) EvaluatesIn.constant
  apply BlockReturnsIn.for_sequence_name rfl (items := ints xs)
  · simp [Mutable.sequence, found]
  · let env := HeapCalls.setLocal
      (HeapCalls.argumentEnv [{ name := "xs", annotation := some (.subscript (.name "list") (.name "int")) }]
        [.ref address]) "m" (.scalar (.int 0))
    exact maximum.loop xs [] heap address found env 0 rfl


private theorem lookup_extended {before after : Mutable.Heap} {address : Nat} {object : Mutable.Object} (selected : before <+: after)
    (found : before[address]? = some object) : after[address]? = some object := by
  obtain ⟨extra, rfl⟩ := selected
  have valid : address < before.length := List.getElem?_eq_some_iff.mp found |>.1
  simpa [List.getElem?_append, valid] using found

private theorem window.empty_test (env : HeapCalls.Env) (xs : List Int) (heap : Mutable.Heap) (address : Nat)
    (lookup : env "xs" = some (.ref address)) (found : heap[address]? = some (.list (ints xs))) :
    EvaluatesIn program env (.unaryOp .not (.name "xs")) heap 3 (.scalar (.bool xs.isEmpty)) heap := by
  apply EvaluatesIn.unary (EvaluatesIn.lookup lookup)
  simp [HeapCalls.unary, Mutable.truth, found, ints]

private theorem window.short_test (env : HeapCalls.Env) (xs : List Int) (k : Int)
    (heap : Mutable.Heap) (address : Nat)
    (lookup : env "xs" = some (.ref address)) (bound : env "k" = some (.scalar (.int k)))
    (found : heap[address]? = some (.list (ints xs))) :
    EvaluatesIn program env (.compare (.builtinCall .len [.name "xs"]) [(.lt, .name "k")])
      heap 8 (.scalar (.bool (decide ((xs.length : Int) < k)))) heap := by
  apply EvaluatesIn.compare (ca := 4) (cb := 1) (av := .scalar (.int xs.length)) (middle := heap) (afterRight := heap)
  · exact EvaluatesIn.builtin (.cons (EvaluatesIn.lookup lookup) .nil)
      (by simp [HeapCalls.primitive, Mutable.length, found, ints])
  · exact EvaluatesIn.lookup bound
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]


/-- Exact weighted prefix cost of the actual window builder. -/
def windowCost (k : Int) : List Int → Nat
  | [] => 9
  | head :: tail =>
      if ((head :: tail).length : Int) < k then 19
      else 8 * ((head :: tail).take
        (FormalSW.Py.Builtin.Slice.prefixLength (head :: tail).length k)).length +
        tail.length + (windowValues k tail).length + windowCost k tail + 61

/-- Weighted execution of recursive windows, with its actual allocated heap. -/
theorem window_products.exact (k : Int) (xs : List Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (ints xs))) :
    ∃ (finalHeap : Mutable.Heap) (answer : Nat), heap <+: finalHeap ∧
      finalHeap[answer]? = some (.list (ints (windowValues k xs))) ∧
      ReturnsIn program "window_products" [.scalar (.int k), .ref address] heap
        (windowCost k xs) (.ref answer) finalHeap := by
  induction xs generalizing heap address with
  | nil =>
      refine ⟨heap ++ [.list []], heap.length, List.prefix_append _ _, ?_, ?_⟩
      · simp [windowValues, ints]
      · apply BlockReturnsIn.function rfl rfl
        apply BlockReturnsIn.branch (cb := 3) (window.empty_test _ [] heap address rfl found) (flag := true)
        · rfl
        · exact BlockReturnsIn.return_value (EvaluatesIn.list .nil)
  | cons head tail ih =>
      by_cases short : (tail.length : Int) + 1 < k
      · refine ⟨heap ++ [.list []], heap.length, List.prefix_append _ _, ?_, ?_⟩
        · simp [windowValues, short, ints]
        · simp only [windowCost, List.length_cons, Int.natCast_add, Int.natCast_one, short, ↓reduceIte]
          apply BlockReturnsIn.function rfl rfl
          apply BlockReturnsIn.branch (cb := 13) (window.empty_test _ (head :: tail) heap address rfl found) (flag := false)
          · rfl
          · apply BlockReturnsIn.branch (cb := 3) (window.short_test _ (head :: tail) k heap address rfl rfl found) (flag := true)
            · simp [Mutable.truth, scalarTruth, short]
            · exact BlockReturnsIn.return_value (EvaluatesIn.list .nil)
      · let selected := (head :: tail).take (FormalSW.Py.Builtin.Slice.prefixLength (head :: tail).length k)
        let product := selected.prod
        let h1 := heap ++ [.list (ints selected)]
        let h2 := h1 ++ [.list [.scalar (.int product)]]
        let h3 := h2 ++ [.list (ints tail)]
        have extends1 : heap <+: h1 := List.prefix_append _ _
        have extends2 : heap <+: h2 := extends1.trans (List.prefix_append _ _)
        have extends3 : heap <+: h3 := extends2.trans (List.prefix_append _ _)
        have prefixFound : h1[heap.length]? = some (.list (ints selected)) := by simp [h1]
        have firstFound : h2[h1.length]? = some (.list [.scalar (.int product)]) := by simp [h2]
        have tailFound : h3[h2.length]? = some (.list (ints tail)) := by simp [h3]
        obtain ⟨last, result, extension, resultFound, recursive⟩ := ih h3 h2.length tailFound
        have firstLast : last[h1.length]? = some (.list [.scalar (.int product)]) :=
          lookup_extended ((List.prefix_append h2 [.list (ints tail)]).trans extension) firstFound
        have executed : ∃ cost, ReturnsIn program "window_products" [.scalar (.int k), .ref address] heap
            cost (.ref last.length) (last ++ [.list (ints (product :: windowValues k tail))]) ∧
            cost = 8 * selected.length + tail.length + (windowValues k tail).length + windowCost k tail + 61 := by
          refine ⟨_, ?_, ?_⟩
          · apply BlockReturnsIn.function rfl rfl
            apply BlockReturnsIn.branch (window.empty_test _ (head :: tail) heap address rfl found) (flag := false)
            · rfl
            · apply BlockReturnsIn.branch (window.short_test _ (head :: tail) k heap address rfl rfl found) (flag := false)
              · simp [Mutable.truth, scalarTruth, short]
              · apply BlockReturnsIn.return_value
                apply EvaluatesIn.binary (av := .ref h1.length) (bv := .ref result)
                  (middle := h2) (afterRight := last) (delta := 1 + (windowValues k tail).length)
                · apply EvaluatesIn.list
                  apply EvaluatesIn.Arguments.cons (middle := h1)
                  · apply EvaluatesIn.call (values := [.ref heap.length]) (middle := h1)
                    · apply EvaluatesIn.Arguments.cons (middle := h1)
                      · apply EvaluatesIn.slice (delta := selected.length)
                          (.cons (EvaluatesIn.lookup rfl) (.cons EvaluatesIn.constant
                            (.cons (EvaluatesIn.lookup rfl) (.cons EvaluatesIn.constant .nil))))
                        · simpa [h1, selected, ints, List.map_take] using Mutable.getSlice_list_prefix heap address (ints (head :: tail)) k found
                        · simp [h1, ints]
                      · exact .nil
                    · exact list_product.result selected h1 heap.length prefixFound
                  · exact .nil
                · apply EvaluatesIn.call (values := [.scalar (.int k), .ref h2.length]) (middle := h3)
                  · apply EvaluatesIn.Arguments.cons (EvaluatesIn.lookup rfl)
                    apply EvaluatesIn.Arguments.cons
                    · apply EvaluatesIn.slice (delta := tail.length)
                        (.cons (EvaluatesIn.lookup rfl) (.cons EvaluatesIn.constant
                          (.cons EvaluatesIn.constant (.cons EvaluatesIn.constant .nil))))
                      · simpa [h3, ints] using Mutable.getSlice_list_suffix h2 address (ints (head :: tail)) 1
                          (lookup_extended extends2 found)
                      · simp [h3, ints]
                    · exact .nil
                  · exact recursive
                · simp [Mutable.binary, show (BinOp.add == BinOp.add) = true from rfl, firstLast, resultFound, Mutable.array, Mutable.allocate, ints, Except.map]
                · simp [ints, Nat.add_comm]
          · omega
        obtain ⟨_, run, rfl⟩ := executed
        refine ⟨last ++ [.list (ints (product :: windowValues k tail))], last.length,
          (extends3.trans extension).trans (List.prefix_append _ _), ?_, ?_⟩
        · simp [windowValues, short, product, selected]
        · convert run using 1 <;>
            simp [windowCost, short, selected, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm,
              Nat.mul_succ, Nat.succ_mul]

end solvers.«8».referenceCost
