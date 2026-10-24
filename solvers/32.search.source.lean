import solvers.«32.outer.source»

set_option Elab.async false

namespace solvers.«32».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.PandigitalProductSearch
private abbrev program := Solvers.P32Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "solve").getD ([], [])).2

def readyEnv (env : HeapCalls.Env) (width address : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal env "product_width" (number (width / 2)))
    "lower" (number (lowerProduct width))) "upper" (number (upperProduct width)))
    "products" (.ref address)

theorem setup_result (width : Nat) (large : 3 ≤ width) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getWidth : env "width" = some (number width)) :
    BlockContinues program env (code.drop 2 |>.take 4) heap (readyEnv env width heap.length)
      (working heap []) := by
  let e1 := setLocal env "product_width" (number (width / 2))
  let e2 := setLocal e1 "lower" (number (lowerProduct width))
  let e3 := setLocal e2 "upper" (number (upperProduct width))
  have exponentPositive : 1 ≤ width / 2 := by omega
  have powerPositive : 1 ≤ 10 ^ (width / 2) := Nat.one_le_pow _ _ (by decide)
  have first : BlockContinues program env [code[2]'(by decide)] heap e1 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number width) (bv := number 2)
      (middle := heap) (afterRight := heap) (Evaluates.lookup getWidth) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary])
  have second : BlockContinues program e1 [code[3]'(by decide)] heap e2 heap := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number 10) (bv := number (width / 2 - 1))
      (middle := heap) (afterRight := heap) Evaluates.constant
    · exact Evaluates.binary (av := number (width / 2)) (bv := number 1)
        (middle := heap) (afterRight := heap) (finalHeap := heap)
        (Evaluates.lookup (by simp [e1, setLocal])) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, Int.natCast_sub exponentPositive])
    · simp [Mutable.binary, scalarBinary, lowerProduct,
        show ¬ ((width / 2 - 1 : Nat) : Int) < 0 by omega]
  have third : BlockContinues program e2 [code[4]'(by decide)] heap e3 heap := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number (10 ^ (width / 2))) (bv := number 1)
      (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (av := number 10) (bv := number (width / 2))
        (middle := heap) (afterRight := heap) (finalHeap := heap) Evaluates.constant
        (Evaluates.lookup (by simp [e2, e1, setLocal]))
        (by simp [Mutable.binary, scalarBinary, show ¬ (width : Int) / 2 < 0 by omega]; rfl)
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, upperProduct, Int.natCast_sub powerPositive]
  have fourth : BlockContinues program e3 [code[5]'(by decide)] heap
      (readyEnv env width heap.length) (working heap []) :=
    BlockContinues.annAssign (Evaluates.builtin .nil rfl)
  exact first.append (second.append (third.append fourth))

/-- The complete factor traversal finishes with the exact set, then sums in every possible order. -/
theorem result (width : Nat) (large : 3 ≤ width) (small : width ≤ 9) (heap : Mutable.Heap) :
    ∃ finalHeap, SetCalls.ReturnsValue program "solve" [number width] heap
      (number (answer width)) finalHeap := by
  let entry := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number width] []
  let ready := readyEnv entry width heap.length
  obtain ⟨finalEnv, scanned, stable⟩ := outer_loop width 1 (Nat.sqrt (upperProduct width))
    (Nat.sqrt (upperProduct width) + 1) (by decide) (by omega) heap [] ready
    (by simp [ready, readyEnv, setLocal, entry]; rfl)
    (by simp [ready, readyEnv, setLocal]) (by simp [ready, readyEnv, setLocal])
    (by simp [ready, readyEnv, setLocal])
  change BlockContinues program ready [.rangeStepLoop "a" 1 (Nat.sqrt (upperProduct width) + 1) 1 outerBody []]
    (working heap []) finalEnv (working heap (inserted [] (candidates width))) at scanned
  have rootEq : Math.isqrt (upperProduct width) = Nat.sqrt (upperProduct width) :=
    Nat.eq_sqrt.mpr (Math.isqrt_spec _)
  have upperArg : Evaluates program ready
      (.binOp (.call (.name "isqrt") [.name "upper"]) .add (.constant (.int 1)))
      (working heap []) (number (Nat.sqrt (upperProduct width) + 1)) (working heap []) := by
    apply Evaluates.binary (av := number (Nat.sqrt (upperProduct width))) (bv := number 1)
      (middle := working heap []) (afterRight := working heap [])
    · apply Evaluates.library_call (.cons (Evaluates.lookup (value := number (upperProduct width))
        (by simp [ready, readyEnv, setLocal])) .nil) rfl
      change scalarResult (some (Math.isqrtCall [.int (upperProduct width)])) (working heap []) = _
      simp [Math.isqrtCall, scalarResult, rootEq, show ¬ (upperProduct width : Int) < 0 by omega]
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary]
  have iterated : BlockContinues program ready [code[6]'(by decide)]
      (working heap []) finalEnv (working heap (inserted [] (candidates width))) :=
    BlockContinues.for_range (.cons Evaluates.constant (.cons upperArg .nil)) rfl scanned
  have exactSet : inserted [] (candidates width) =
      ((candidates width).eraseDups.map fun (n : Nat) => Constant.int n) := by
    simpa only [inserted] using ScalarSet.fold_insert_nats (candidates width)
  have guarded : BlockContinues program entry (code.take 2) heap entry heap := by
    have low : BlockContinues program entry [code[0]'(by decide)] heap entry heap := by
      apply BlockContinues.branch (middle := heap) (flag := false) (value := flag false)
        (Evaluates.compare (av := number width) (bv := number 3) (middle := heap) (afterRight := heap)
          (Evaluates.lookup rfl) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
      exact .nil
    have high : BlockContinues program entry [code[1]'(by decide)] heap entry heap := by
      apply BlockContinues.branch (middle := heap) (flag := false) (value := flag false)
        (Evaluates.compare (av := number width) (bv := number 9) (middle := heap) (afterRight := heap)
          (Evaluates.lookup rfl) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
      exact .nil
    exact low.append high
  have getProducts : finalEnv "products" = some (.ref heap.length) :=
    (stable "products" (by decide)).trans (by simp [ready, readyEnv, setLocal])
  have sumCast (items : List Nat) : (items.map (fun (n : Nat) => (n : Int))).sum = (items.sum : Int) := by
    induction items with
    | nil => rfl
    | cons n rest ih => simp [ih]
  have found : (working heap (inserted [] (candidates width)))[heap.length]? =
      some (.set ((((candidates width).eraseDups).map (fun (n : Nat) => (n : Int))).map Constant.int)) := by
    simpa only [exactSet, List.map_map, Function.comp_def] using
      working_found heap (inserted [] (candidates width))
  have summed := SetCalls.Evaluates.sum_integers (items := (candidates width).eraseDups.map (fun (n : Nat) => (n : Int)))
    (initial := 0) (module := program) (env := finalEnv) (globals := [])
    (iterable := .name "products") (start := .constant (.int 0))
    (.cons (Evaluates.lookup getProducts) (.cons Evaluates.constant .nil)) found
  refine ⟨working heap (inserted [] (candidates width)), ?_⟩
  apply SetCalls.BlockReturns.function rfl rfl
  apply SetCalls.BlockReturns.after_heap guarded
  apply SetCalls.BlockReturns.after_heap (setup_result width large heap entry rfl)
  apply SetCalls.BlockReturns.after_heap iterated
  apply SetCalls.BlockReturns.return_value
  simpa only [Int.zero_add, sumCast, answer] using summed


theorem result_int (width : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, SetCalls.ReturnsValue program "solve" [.scalar (.int width)] heap
      (number (answer width.toNat)) finalHeap := by
  by_cases low : width < 3
  · refine ⟨heap, SetCalls.ReturnsValue.of_heap ?_⟩
    rw [answer_small width.toNat (by omega)]
    apply HeapCalls.BlockReturns.function rfl rfl
    apply HeapCalls.BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
    · exact Evaluates.compare (av := .scalar (.int width)) (bv := number 3)
        (middle := heap) (afterRight := heap) (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, low])
    · rfl
    · exact HeapCalls.BlockReturns.return_value Evaluates.constant
  · by_cases high : 9 < width
    · refine ⟨heap, SetCalls.ReturnsValue.of_heap ?_⟩
      rw [answer_large width.toNat (by omega)]
      apply HeapCalls.BlockReturns.function rfl rfl
      apply HeapCalls.BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
      · exact Evaluates.compare (av := .scalar (.int width)) (bv := number 3)
          (middle := heap) (afterRight := heap) (Evaluates.lookup rfl) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, low])
      · rfl
      · apply HeapCalls.BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
        · exact Evaluates.compare (av := .scalar (.int width)) (bv := number 9)
            (middle := heap) (afterRight := heap) (Evaluates.lookup rfl) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, high])
        · rfl
        · exact HeapCalls.BlockReturns.return_value Evaluates.constant
    · lift width to Nat using (by omega : 0 ≤ width)
      exact result width (by omega) (by omega) heap

end solvers.«32».solve
