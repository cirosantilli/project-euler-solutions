import «.formalsw».solvers.«34.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerStatements.Lib.DigitFactorial
import data.«project-euler-statements».data.py.project_euler_statements.«p34.py»

namespace solvers.«34»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P34

private abbrev program := Solvers.P34Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev encode (items : List Nat) : List Mutable.Value := items.map number
private def factorialTable : List Mutable.Value := encode ((List.range 10).map Nat.factorial)

set_option Elab.async false

private theorem found_append {heap : Mutable.Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (extra : Mutable.Heap) :
    (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private def tableBefore (next : Nat) : List Mutable.Value :=
  encode ((List.range 10).map (fun d => if d < next then Nat.factorial d else 1))

private theorem table_length (next : Nat) : (tableBefore next).length = 10 := by
  simp [tableBefore, encode]

private theorem table_initial : tableBefore 1 = List.replicate 10 (number 1) := by decide +kernel
private theorem table_complete : tableBefore 10 = factorialTable := by decide +kernel
private theorem table_update_checked : ∀ index : Fin 9,
    (tableBefore (index.val + 1)).set (index.val + 1) (number (Nat.factorial (index.val + 1))) =
      tableBefore (index.val + 2) := by decide +kernel

private theorem table_update (next : Nat) (lower : 1 ≤ next) (upper : next < 10) :
    (tableBefore next).set next (number (Nat.factorial next)) = tableBefore (next + 1) := by
  have checked := table_update_checked ⟨next - 1, by omega⟩
  have current : next - 1 + 1 = next := by omega
  have following : next - 1 + 2 = next + 1 := by omega
  simpa only [current, following] using checked

private abbrev fillBody : List Stmt :=
  [.augAssign (.name "f") .mult (.name "d"),
   .assign [.subscript (.name "facts") (.name "d")] (.name "f")]

private def tableHeap (base : Mutable.Heap) (next : Nat) : Mutable.Heap :=
  base ++ [.list (tableBefore next)]

private theorem fill_step (next : Nat) (lower : 1 ≤ next) (upper : next < 10)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (getFacts : env "facts" = some (.ref base.length))
    (getDigit : env "d" = some (number next))
    (getFactorial : env "f" = some (number (Nat.factorial (next - 1)))) :
    BlockContinues program env fillBody (tableHeap base next)
      (setLocal env "f" (number (Nat.factorial next))) (tableHeap base (next + 1)) := by
  have product : Nat.factorial (next - 1) * next = Nat.factorial next := by
    cases next with
    | zero => omega
    | succ next => simp [Nat.factorial_succ, Nat.mul_comm]
  apply BlockContinues.append
    (BlockContinues.augAssign (left := number (Nat.factorial (next - 1)))
      (right := number next) (value := number (Nat.factorial next))
      (middle := tableHeap base next) (finalHeap := tableHeap base next)
      getFactorial (Evaluates.lookup getDigit)
      (by simp [inplace, Mutable.binary, scalarBinary, number, ← product]))
  apply BlockContinues.setItem (value := number (Nat.factorial next))
    (by intros; simp) (Evaluates.lookup (by simp [setLocal]))
    (.cons (Evaluates.lookup (by simpa [setLocal] using getFacts))
      (.cons (Evaluates.lookup (by simpa [setLocal] using getDigit)) .nil))
  simp [Mutable.setItem, Mutable.index, tableHeap, table_length, number,
    show ¬(next : Int) < 0 by omega, show ¬10 ≤ next by omega, Except.map,
    List.set_append_right, table_update next lower upper]

private theorem fill_loop (next remaining : Nat) (lower : 1 ≤ next) (length : next + remaining = 10)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (getFacts : env "facts" = some (.ref base.length))
    (getFactorial : env "f" = some (number (Nat.factorial (next - 1)))) :
    ∃ finalEnv,
      BlockContinues program env [.rangeStepLoop "d" next 10 1 fillBody []]
        (tableHeap base next) finalEnv (base ++ [.list factorialTable]) ∧
      finalEnv "facts" = some (.ref base.length) := by
  cases remaining with
  | zero =>
      have finished : next = 10 := by omega
      subst next
      refine ⟨env, ?_, getFacts⟩
      rw [tableHeap, table_complete]
      exact BlockContinues.range_done rfl .nil
  | succ remaining =>
      let bound := setLocal env "d" (number next)
      let updated := setLocal bound "f" (number (Nat.factorial next))
      obtain ⟨finalEnv, rest, finalFacts⟩ := fill_loop (next + 1) remaining (by omega) (by omega)
        base updated (by simpa [updated, bound, setLocal] using getFacts)
        (by simp [updated, setLocal])
      refine ⟨finalEnv, ?_, finalFacts⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have filled := fill_step next lower (by omega) base bound
        (by simpa [bound, setLocal] using getFacts) (by simp [bound, setLocal])
        (by simpa [bound, setLocal] using getFactorial)
      simpa only [Calls.loopContinuation, show Calls.loopControl fillBody = false from rfl,
        ↓reduceIte, Int.natCast_add, Int.natCast_one] using filled.append rest
termination_by remaining

private theorem initialize_table (env : HeapCalls.Env) (heap : Mutable.Heap) :
    Evaluates program env (.binOp (.list [.constant (.int 1)]) .mult (.constant (.int 10)))
      heap (.ref (heap.length + 1)) (tableHeap (heap ++ [.list [number 1]]) 1) := by
  let base := heap ++ [.list [number 1]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number 10)
    (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.constant
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, base, tableHeap, table_initial, Except.map]

/-- Source repetition allocates its one-element operand and the mutable table.
The nine source assignments fill the latter while retaining its identity. -/
theorem precompute_factorials.result (heap : Mutable.Heap) :
    ReturnsValue program "precompute_factorials" [] heap (.ref (heap.length + 1))
      (heap ++ [.list [number 1], .list factorialTable]) := by
  let args := ((findFunctionIn program.body "precompute_factorials").getD ([], [])).1
  let body := ((findFunctionIn program.body "precompute_factorials").getD ([], [])).2
  let initial := functionEnv args body [] []
  let ready := setLocal (setLocal initial "facts" (.ref (heap.length + 1))) "f" (number 1)
  let base := heap ++ [.list [number 1]]
  obtain ⟨finalEnv, filled, finalFacts⟩ := fill_loop 1 9 (by decide) rfl base ready
    (by simp [base, ready, setLocal]) (by simp [ready, setLocal])
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (initialize_table initial heap)
  apply BlockReturns.assign Evaluates.constant
  have loop : BlockContinues program ready
      [.forIn (.name "d") (.call (.name "range") [.constant (.int 1), .constant (.int 10)]) fillBody []]
      (tableHeap base 1) finalEnv (base ++ [.list factorialTable]) :=
    BlockContinues.for_range (.cons Evaluates.constant (.cons Evaluates.constant .nil)) rfl filled
  have returned := loop.returns (BlockReturns.return_value (rest := []) (Evaluates.lookup finalFacts))
  simpa only [base, List.append_assoc, List.singleton_append, List.length_append,
    List.length_singleton] using returned


private theorem read_factorial (heap : Mutable.Heap) (address d : Nat) (inside : d < 10)
    (found : heap[address]? = some (.list factorialTable)) (env : HeapCalls.Env)
    (getFacts : env "facts" = some (.ref address))
    (getRemainder : env "r" = some (number d)) :
    Evaluates program env (.subscript (.name "facts") (.name "r")) heap
      (number (Nat.factorial d)) heap := by
  apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup getFacts) (Evaluates.lookup getRemainder)
  have entry : factorialTable[d]? = some (number (Nat.factorial d)) := by simp [factorialTable, encode, inside]
  simpa only [(List.getElem?_eq_some_iff.mp entry).2] using
    Mutable.getItem_list heap address d factorialTable found (by simpa [factorialTable, encode] using inside)


private abbrev digitBody : List Stmt :=
  [.assign [.tuple [.name "n", .name "r"]] (.builtinCall .divmod [.name "n", .constant (.int 10)]),
   .augAssign (.name "s") .add (.subscript (.name "facts") (.name "r"))]
private abbrev digitWhile : Stmt :=
  .whileDo (.name "n") digitBody []

/-- Divmod creates temporary tuples, but never changes the input factorial table. -/
private theorem digit_loop (n total address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list factorialTable))
    (getN : env "n" = some (number n)) (getS : env "s" = some (number total))
    (getFacts : env "facts" = some (.ref address)) :
    ∃ extra, BlockReturns program env [digitWhile, .return (.name "s")] heap
      (number (total + digitFactSum n)) (heap ++ extra) := by
  by_cases positive : 0 < n
  · let pair : Mutable.Object := .tuple [number (n / 10), number (n % 10)]
    let unpacked := assignNames env ["n", "r"] [number (n / 10), number (n % 10)]
    let updated := setLocal unpacked "s" (number (total + Nat.factorial (n % 10)))
    obtain ⟨extra, rest⟩ := digit_loop (n / 10) (total + Nat.factorial (n % 10)) address
      (heap ++ [pair]) updated (found_append found [pair])
      (by simp [updated, unpacked, assignNames, setLocal]) (by simp [updated, setLocal])
      (by simpa [updated, unpacked, assignNames, setLocal] using getFacts)
    refine ⟨[pair] ++ extra, ?_⟩
    rw [← List.append_assoc]
    apply BlockReturns.while_loop (flag := true)
      (Evaluates.lookup getN)
      (by simp [Mutable.truth, scalarTruth]; omega)
    apply BlockReturns.unpack (items := [number (n / 10), number (n % 10)])
      (afterUnpack := heap ++ [pair]) rfl
    · apply Evaluates.builtin (.cons (Evaluates.lookup getN) (.cons Evaluates.constant .nil))
      have divided := Mutable.divmod_ints (n : Int) 10 (by decide) heap
      have quotient : (n : Int).fdiv 10 = ((n / 10 : Nat) : Int) := by
        rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide)]; rfl
      have remainder : (n : Int) - 10 * (n : Int).fdiv 10 = ((n % 10 : Nat) : Int) := by
        rw [Int.fdiv_eq_ediv_of_nonneg _ (by decide), ← Int.emod_def]; rfl
      simpa only [quotient, remainder] using divided
    · simp [Mutable.sequence, pair, Int.emod_def]
    · rfl
    · apply BlockReturns.augAssign (left := number total) (right := number (Nat.factorial (n % 10)))
        (value := number (total + Nat.factorial (n % 10))) (middle := heap ++ [pair])
        (afterOperation := heap ++ [pair])
        (by simpa [assignNames, setLocal] using getS)
      · exact read_factorial _ address (n % 10) (Nat.mod_lt _ (by decide))
          (found_append found [pair]) _
          (by simpa [assignNames, setLocal] using getFacts) (by simp [assignNames, setLocal])
      · simp [inplace, Mutable.binary, scalarBinary, number]
      · simpa [digitWhile, digitBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          updated, unpacked, ProjectEulerStatements.DigitFactorial.digitFactSum_step n positive, Nat.add_assoc] using rest
  · have zero : n = 0 := by omega
    subst n
    refine ⟨[], ?_⟩
    simp only [List.append_nil]
    apply BlockReturns.while_loop (flag := false)
      (Evaluates.lookup getN) rfl
    simpa [ProjectEulerStatements.DigitFactorial.digitFactSum_zero] using BlockReturns.return_value (module := program) (Evaluates.lookup getS)
termination_by n
decreasing_by exact Nat.div_lt_self positive (by decide)

theorem digit_factorial_sum.result (n address : Nat) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list factorialTable)) :
    ∃ extra, ReturnsValue program "digit_factorial_sum" [number n, .ref address] heap
      (number (digitFactSum n)) (heap ++ extra) := by
  obtain ⟨extra, loop⟩ := digit_loop n 0 address heap
    (setLocal (functionEnv ((findFunctionIn program.body "digit_factorial_sum").getD ([], [])).1
      ((findFunctionIn program.body "digit_factorial_sum").getD ([], [])).2 [number n, .ref address] [])
      "s" (number 0)) found rfl rfl rfl
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  simpa only [Nat.zero_add] using loop



private abbrev boundCondition : Expr :=
  .compare (.binOp (.name "n") .mult (.name "nine_fact"))
    [(.gtE, .binOp (.constant (.int 10)) .pow
      (.binOp (.name "n") .sub (.constant (.int 1))))]

private abbrev boundBody : List Stmt := [.augAssign (.name "n") .add (.constant (.int 1))]
private abbrev boundWhile : Stmt := .whileDo boundCondition boundBody []

private theorem bound_guard (n : Nat) (positive : 1 ≤ n)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n))
    (getNine : env "nine_fact" = some (number (Nat.factorial 9))) :
    Evaluates program env boundCondition heap
      (.scalar (.bool (decide (10 ^ (n - 1) ≤ n * Nat.factorial 9)))) heap := by
  apply Evaluates.compare (av := number (n * Nat.factorial 9)) (bv := number (10 ^ (n - 1)))
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (Evaluates.lookup getN) (Evaluates.lookup getNine)
      (by simp [Mutable.binary, scalarBinary, number])
  · apply Evaluates.binary (av := number 10) (bv := number (n - 1))
      (middle := heap) (afterRight := heap) Evaluates.constant
    · exact Evaluates.binary (Evaluates.lookup getN) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub positive])
    · have nonnegative : ¬ ((n - 1 : Nat) : Int) < 0 := by omega
      simp [Mutable.binary, scalarBinary, number, nonnegative]
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    exact Int.ofNat_le

private theorem bound_loop (n remaining : Nat) (positive : 1 ≤ n) (length : n + remaining = 8)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n))
    (getNine : env "nine_fact" = some (number (Nat.factorial 9))) :
    ∃ finalEnv, BlockContinues program env [boundWhile] heap finalEnv heap ∧
      finalEnv "n" = some (number 8) ∧
      (∀ name, name ≠ "n" → finalEnv name = env name) := by
  cases remaining with
  | zero =>
      have doneN : n = 8 := by omega
      subst n
      refine ⟨env, ?_, getN, fun _ _ => rfl⟩
      have evaluated := bound_guard 8 (by decide) env heap getN getNine
      have failed : decide (10 ^ (8 - 1) ≤ 8 * Nat.factorial 9) = false := by decide +kernel
      rw [failed] at evaluated
      exact BlockContinues.while_loop evaluated rfl .nil
  | succ remaining =>
      have small : n < 8 := by omega
      have member : n ∈ List.range' 1 7 := List.mem_range'_1.mpr ⟨positive, by omega⟩
      have active : decide (10 ^ (n - 1) ≤ n * Nat.factorial 9) = true :=
        List.all_eq_true.mp ProjectEulerStatements.DigitFactorial.digit_bound_guards.1 n member
      let advanced := setLocal env "n" (number (n + 1))
      obtain ⟨finalEnv, rest, finalN, preserved⟩ := bound_loop (n + 1) remaining (by omega) (by omega)
        advanced heap (by simp [advanced, setLocal]) (by simpa [advanced, setLocal] using getNine)
      refine ⟨finalEnv, ?_, finalN, ?_⟩
      · have evaluated := bound_guard n positive env heap getN getNine
        rw [active] at evaluated
        apply BlockContinues.while_loop evaluated rfl
        have increment : BlockContinues program env boundBody heap advanced heap :=
          BlockContinues.augAssign (left := number n) (right := number 1)
            (value := number (n + 1)) (middle := heap) (finalHeap := heap) getN Evaluates.constant
            (by simp [inplace, Mutable.binary, scalarBinary, number])
        simpa only [Calls.loopContinuation, show Calls.loopControl boundBody = false from rfl,
          ↓reduceIte] using increment.append rest
      · intro name other
        rw [preserved name other]
        simp [advanced, setLocal, other]
termination_by remaining


private def scanValue (items : List Nat) (total : Nat) : Nat :=
  items.foldl (fun acc n => if digitFactSum n = n then acc + n else acc) total

private abbrev scanBody : List Stmt :=
  [.ifThen (.compare (.name "x")
    [(.eq, .call (.name "digit_factorial_sum") [.name "x", .name "facts"])])
    [.augAssign (.name "total") .add (.name "x")] []]

/- This induction is over an arbitrary finite interval. The default call uses
the theorem as an opaque proof; elaboration never evaluates its 2.54M iterations. -/
private theorem scan (first count stop total address : Nat) (length : first + count = stop)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list factorialTable))
    (getFacts : env "facts" = some (.ref address))
    (getTotal : env "total" = some (number total)) :
    ∃ extra finalEnv,
      BlockContinues program env [.rangeStepLoop "x" first stop 1 scanBody []]
        heap finalEnv (heap ++ extra) ∧
      finalEnv "facts" = some (.ref address) ∧
      finalEnv "total" = some (number (scanValue (List.range' first count) total)) := by
  induction count generalizing first total heap env with
  | zero =>
      refine ⟨[], env, ?_, getFacts, ?_⟩
      · simpa only [List.append_nil] using
          (BlockContinues.range_done (module := program) (heap := heap)
            (env := env) (by simp [Calls.rangeContinues]; omega) BlockContinues.nil)
      · simpa [scanValue] using getTotal
  | succ count ih =>
      let bound := setLocal env "x" (number first)
      have boundFacts : bound "facts" = some (.ref address) := by simpa [bound, setLocal] using getFacts
      have boundTotal : bound "total" = some (number total) := by simpa [bound, setLocal] using getTotal
      have boundX : bound "x" = some (number first) := by simp [bound, setLocal]
      obtain ⟨extra, helper⟩ := digit_factorial_sum.result first address heap found
      have preserved := found_append found extra
      have evaluated : Evaluates program bound
          (.compare (.name "x") [(.eq, .call (.name "digit_factorial_sum") [.name "x", .name "facts"])])
          heap (.scalar (.bool (decide (digitFactSum first = first)))) (heap ++ extra) := by
        apply Evaluates.compare (av := number first) (bv := number (digitFactSum first))
          (middle := heap) (afterRight := heap ++ extra) (Evaluates.lookup boundX)
          (Evaluates.call (.cons (Evaluates.lookup boundX) (.cons (Evaluates.lookup boundFacts) .nil)) helper)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, eq_comm]
      by_cases accepted : digitFactSum first = first
      · let updated := setLocal bound "total" (number (total + first))
        obtain ⟨later, finalEnv, rest, finalFacts, finalTotal⟩ := ih (first + 1) (total + first)
          (by omega) (heap ++ extra) updated preserved
          (by simpa [updated, setLocal] using boundFacts) (by simp [updated, setLocal])
        refine ⟨extra ++ later, finalEnv, ?_, finalFacts, ?_⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          have increment : BlockContinues program bound
              [.augAssign (.name "total") .add (.name "x")] (heap ++ extra) updated (heap ++ extra) :=
            BlockContinues.augAssign (left := number total) (right := number first)
              (value := number (total + first)) (middle := heap ++ extra) (finalHeap := heap ++ extra)
              boundTotal (Evaluates.lookup boundX)
              (by simp [inplace, Mutable.binary, scalarBinary, number])
          have iteration : BlockContinues program bound scanBody heap updated (heap ++ extra) :=
            BlockContinues.branch (flag := true) (by simpa [accepted] using evaluated) rfl increment
          simpa only [Calls.loopContinuation, show Calls.loopControl scanBody = false from rfl,
            ↓reduceIte, Int.natCast_add, Int.natCast_one, List.append_assoc] using iteration.append rest
        · simpa [scanValue, List.range'_succ, accepted] using finalTotal
      · obtain ⟨later, finalEnv, rest, finalFacts, finalTotal⟩ := ih (first + 1) total (by omega)
          (heap ++ extra) bound preserved boundFacts boundTotal
        refine ⟨extra ++ later, finalEnv, ?_, finalFacts, ?_⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          have iteration : BlockContinues program bound scanBody heap bound (heap ++ extra) :=
            BlockContinues.branch (by simpa [accepted] using evaluated) rfl .nil
          simpa only [Calls.loopContinuation, show Calls.loopControl scanBody = false from rfl,
            ↓reduceIte, Int.natCast_add, Int.natCast_one, List.append_assoc] using iteration.append rest
        · simpa [scanValue, List.range'_succ, accepted] using finalTotal


theorem solve.result :
    ∃ finalHeap, ReturnsValue program "solve" [] [] (number (naive 2540160)) finalHeap := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let initial := functionEnv args body [] []
  let heap : Mutable.Heap := [.list [number 1], .list factorialTable]
  let factsEnv := setLocal initial "facts" (.ref 1)
  let nineEnv := setLocal factsEnv "nine_fact" (number (Nat.factorial 9))
  let nEnv := setLocal nineEnv "n" (number 1)
  obtain ⟨afterBound, boundLoop, boundN, preserved⟩ := bound_loop 1 7 (by decide) rfl nEnv heap
    (by simp [nEnv, setLocal]) (by simp [nEnv, nineEnv, setLocal])
  have boundFacts : afterBound "facts" = some (.ref 1) := by
    rw [preserved "facts" (by decide)]
    simp [nEnv, nineEnv, factsEnv, setLocal]
  have boundNine : afterBound "nine_fact" = some (number (Nat.factorial 9)) := by
    rw [preserved "nine_fact" (by decide)]
    simp [nEnv, nineEnv, setLocal]
  let upperEnv := setLocal afterBound "upper" (number 2540160)
  let ready := setLocal upperEnv "total" (number 0)
  obtain ⟨extra, finalEnv, scanned, finalFacts, finalTotal⟩ :=
    scan 3 2540158 2540161 0 1 rfl heap ready (by simp [heap])
      (by simpa [ready, upperEnv, setLocal] using boundFacts) (by simp [ready, setLocal])
  have answer : scanValue (List.range' 3 2540158) 0 = naive 2540160 :=
    (ProjectEulerStatements.DigitFactorial.naive_range3 2540160 (by decide)).symm
  rw [answer] at finalTotal
  refine ⟨heap ++ extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (Evaluates.call .nil (precompute_factorials.result []))
  apply BlockReturns.assign (value := number (Nat.factorial 9)) (middle := heap)
  · exact Evaluates.subscript (by intros; simp) (Evaluates.lookup rfl) Evaluates.constant rfl
  · apply BlockReturns.assign Evaluates.constant
    apply boundLoop.returns
    apply BlockReturns.assign (value := number 2540160) (middle := heap)
    · apply Evaluates.binary (av := number 7) (bv := number (Nat.factorial 9))
        (middle := heap) (afterRight := heap)
      · exact Evaluates.binary (Evaluates.lookup boundN) Evaluates.constant rfl
      · exact Evaluates.lookup boundNine
      · rfl
    · apply BlockReturns.assign Evaluates.constant
      have loop : BlockContinues program ready
          [.forIn (.name "x") (.call (.name "range")
            [.constant (.int 3), .binOp (.name "upper") .add (.constant (.int 1))]) scanBody []]
          heap finalEnv (heap ++ extra) := by
        apply BlockContinues.for_range (middle := heap)
          (.cons Evaluates.constant
          (.cons (Evaluates.binary (av := number 2540160) (bv := number 1)
            (value := number 2540161) (middle := heap) (afterRight := heap) (finalHeap := heap)
            (Evaluates.lookup (by simp [ready, upperEnv, setLocal])) Evaluates.constant rfl) .nil)) rfl
        exact scanned
      exact loop.returns (BlockReturns.return_value (Evaluates.lookup finalTotal))

theorem solve.«Equiv(naive_default)» :
    (HeapCalls.scalarFunction program "solve" []).Equiv
      (HeapCalls.scalarFunction P34Ast.module "naive_default" []) := by
  obtain ⟨_, returned⟩ := solve.result
  obtain ⟨_, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p34.naive_default.result []
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := []) returned
  have original := HeapCalls.ReturnsValue.scalar_finishes (values := []) reference
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem solve.«Supported» :
    (HeapCalls.scalarFunction program "solve" []).Supported ∧
      (HeapCalls.scalarFunction P34Ast.module "naive_default" []).Supported := by
  obtain ⟨_, returned⟩ := solve.result
  exact ⟨(HeapCalls.ReturnsValue.scalar_returns (values := []) returned).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p34.naive_default.«Supported»⟩

end solvers.«34»
