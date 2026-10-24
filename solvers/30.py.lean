import «.formalsw».solvers.«30.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p30.py»
import lang.py.FormalSW.SumCalls
import ProjectEulerStatements.Lib.FifthPowerDigits

namespace solvers.«30»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerStatements.P30
private abbrev program := Solvers.P30Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev encode (ns : List Nat) : List Mutable.Value := ns.map number
private def powers : List Mutable.Value := encode ((List.range 10).map (· ^ 5))

set_option Elab.async false

private theorem found_append {heap : Mutable.Heap} {address : Nat} {object : Mutable.Object}
    (found : heap[address]? = some object) (extra : Mutable.Heap) :
    (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left]
  · exact found
  · exact (List.getElem?_eq_some_iff.mp found).1

private theorem read_power (heap : Mutable.Heap) (address d : Nat) (inside : d < 10)
    (found : heap[address]? = some (.list powers)) (env : HeapCalls.Env)
    (getFifth : env "fifth" = some (.ref address))
    (getDigit : env "d" = some (number d)) :
    Evaluates program env (.subscript (.name "fifth") (.name "d")) heap
      (number (d ^ 5)) heap := by
  apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup getFifth) (Evaluates.lookup getDigit)
  have entry : powers[d]? = some (number (d ^ 5)) := by simp [powers, encode, inside]
  simpa only [(List.getElem?_eq_some_iff.mp entry).2] using
    Mutable.getItem_list heap address d powers found (by simpa [powers, encode] using inside)

private theorem digit_step (n : Nat) (positive : 0 < n) :
    digitPowerSum 5 n = (n % 10) ^ 5 + digitPowerSum 5 (n / 10) := by
  simp only [digitPowerSum, Nat.digits_eq_cons_digits_div (by decide : 1 < 10) (by omega : n ≠ 0),
    List.foldl_cons, foldl_add_eq_sum, Nat.zero_add]

private abbrev digitBody : List Stmt :=
  [.assign [.tuple [.name "n", .name "d"]] (.builtinCall .divmod [.name "n", .constant (.int 10)]),
   .augAssign (.name "s") .add (.subscript (.name "fifth") (.name "d"))]
private abbrev digitWhile : Stmt :=
  .whileDo (.compare (.name "n") [(.gt, .constant (.int 0))]) digitBody []

/-- Divmod creates temporary tuples, but never changes the input power table. -/
private theorem digit_loop (n total address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list powers))
    (getN : env "n" = some (number n)) (getS : env "s" = some (number total))
    (getFifth : env "fifth" = some (.ref address)) :
    ∃ extra, BlockReturns program env [digitWhile, .return (.name "s")] heap
      (number (total + digitPowerSum 5 n)) (heap ++ extra) := by
  by_cases positive : 0 < n
  · let pair : Mutable.Object := .tuple [number (n / 10), number (n % 10)]
    let unpacked := assignNames env ["n", "d"] [number (n / 10), number (n % 10)]
    let updated := setLocal unpacked "s" (number (total + (n % 10) ^ 5))
    obtain ⟨extra, rest⟩ := digit_loop (n / 10) (total + (n % 10) ^ 5) address
      (heap ++ [pair]) updated (found_append found [pair])
      (by simp [updated, unpacked, assignNames, setLocal]) (by simp [updated, setLocal])
      (by simpa [updated, unpacked, assignNames, setLocal] using getFifth)
    refine ⟨[pair] ++ extra, ?_⟩
    rw [← List.append_assoc]
    apply BlockReturns.while_loop (flag := true)
      (Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap)
        (Evaluates.lookup getN) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, positive])) rfl
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
    · apply BlockReturns.augAssign (left := number total) (right := number ((n % 10) ^ 5))
        (value := number (total + (n % 10) ^ 5)) (middle := heap ++ [pair])
        (afterOperation := heap ++ [pair])
        (by simpa [assignNames, setLocal] using getS)
      · exact read_power _ address (n % 10) (Nat.mod_lt _ (by decide))
          (found_append found [pair]) _
          (by simpa [assignNames, setLocal] using getFifth) (by simp [assignNames, setLocal])
      · simp [inplace, Mutable.binary, scalarBinary, number]
      · simpa [digitWhile, digitBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          updated, unpacked, digit_step n positive, Nat.add_assoc] using rest
  · have zero : n = 0 := by omega
    subst n
    refine ⟨[], ?_⟩
    simp only [List.append_nil]
    apply BlockReturns.while_loop (flag := false)
      (Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap)
        (Evaluates.lookup getN) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])) rfl
    simpa [digitPowerSum] using BlockReturns.return_value (module := program) (Evaluates.lookup getS)
termination_by n
decreasing_by exact Nat.div_lt_self positive (by decide)

private theorem digit_result (n address : Nat) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list powers)) :
    ∃ extra, ReturnsValue program "digit_fifth_power_sum" [number n, .ref address] heap
      (number (digitPowerSum 5 n)) (heap ++ extra) := by
  obtain ⟨extra, loop⟩ := digit_loop n 0 address heap
    (setLocal (functionEnv ((findFunctionIn program.body "digit_fifth_power_sum").getD ([], [])).1
      ((findFunctionIn program.body "digit_fifth_power_sum").getD ([], [])).2 [number n, .ref address] [])
      "s" (number 0)) found rfl rfl rfl
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  simpa only [Nat.zero_add] using loop

private abbrev selected (n : Nat) : Bool := decide (digitPowerSum 5 n = n)
private abbrev scanBody : List Stmt :=
  [.ifThen (.compare (.name "n") [(.eq, .call (.name "digit_fifth_power_sum") [.name "n", .name "fifth"])])
    [.expr (.call (.attribute (.name "matches") "append") [.name "n"])] []]

/-- The scan preserves both live list addresses while helper calls allocate temporary tuples. -/
private theorem scan (count first powerAddress matchAddress : Nat) (stop : Int)
    (length : (stop - first).toNat = count) (distinct : powerAddress ≠ matchAddress)
    (before : List Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (foundPowers : heap[powerAddress]? = some (.list powers))
    (foundMatches : heap[matchAddress]? = some (.list (encode before)))
    (getFifth : env "fifth" = some (.ref powerAddress))
    (getMatches : env "matches" = some (.ref matchAddress)) :
    ∃ finalHeap finalEnv,
      BlockContinues program env [.rangeStepLoop "n" first stop 1 scanBody []] heap finalEnv finalHeap ∧
      finalHeap[matchAddress]? = some (.list (encode (before ++ (List.range' first count).filter selected))) ∧
      finalEnv "matches" = some (.ref matchAddress) := by
  induction count generalizing first before heap env with
  | zero =>
      refine ⟨heap, env, ?_, ?_, getMatches⟩
      · apply BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
      · simpa using foundMatches
  | succ count ih =>
      let bound := setLocal env "n" (number first)
      have fifth : bound "fifth" = some (.ref powerAddress) := by simpa [bound, setLocal] using getFifth
      have matchBound : bound "matches" = some (.ref matchAddress) := by simpa [bound, setLocal] using getMatches
      obtain ⟨extra, called⟩ := digit_result first powerAddress heap foundPowers
      have foundPowers' := found_append foundPowers extra
      have foundMatches' := found_append foundMatches extra
      have tested : Evaluates program bound
          (.compare (.name "n") [(.eq, .call (.name "digit_fifth_power_sum") [.name "n", .name "fifth"])])
          heap (.scalar (.bool (selected first))) (heap ++ extra) := by
        apply Evaluates.compare (av := number first) (bv := number (digitPowerSum 5 first))
          (middle := heap) (afterRight := heap ++ extra)
          (Evaluates.lookup (by simp [bound, setLocal]))
          (Evaluates.call (.cons (Evaluates.lookup (by simp [bound, setLocal]))
            (.cons (Evaluates.lookup fifth) .nil)) called)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, selected, eq_comm]
      by_cases yes : digitPowerSum 5 first = first
      · let middle := (heap ++ extra).set matchAddress (.list (encode (before ++ [first])))
        have nextPowers : middle[powerAddress]? = some (.list powers) := by
          simpa [middle, List.getElem?_set_ne (Ne.symm distinct)] using foundPowers'
        have nextMatches : middle[matchAddress]? = some (.list (encode (before ++ [first]))) := by
          exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp foundMatches').1
        obtain ⟨finalHeap, finalEnv, rest, finalMatches, finalBound⟩ :=
          ih (first + 1) (by omega) (before ++ [first]) middle bound nextPowers nextMatches fifth matchBound
        refine ⟨finalHeap, finalEnv, ?_, ?_, finalBound⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          apply BlockContinues.append (body := scanBody) (middleEnv := bound) (middleHeap := middle)
          · apply BlockContinues.branch (flag := true) (by simpa [selected, yes] using tested) rfl
            simpa [middle, encode, List.map_append] using
              BlockContinues.discard (Evaluates.append (Evaluates.lookup matchBound) foundMatches'
                (Evaluates.lookup (by simp [bound, setLocal])) foundMatches')
          · simpa [scanBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
              Nat.cast_add, Nat.cast_one] using rest
        · simpa [List.range'_succ, selected, yes, List.append_assoc] using finalMatches
      · obtain ⟨finalHeap, finalEnv, rest, finalMatches, finalBound⟩ :=
          ih (first + 1) (by omega) before (heap ++ extra) bound foundPowers' foundMatches' fifth matchBound
        refine ⟨finalHeap, finalEnv, ?_, ?_, finalBound⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          apply BlockContinues.append (body := scanBody) (middleEnv := bound) (middleHeap := heap ++ extra)
          · exact BlockContinues.branch (flag := false) (by simpa [selected, yes] using tested) rfl .nil
          · simpa [scanBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
              Nat.cast_add, Nat.cast_one] using rest
        · simpa [List.range'_succ, selected, yes] using finalMatches

private abbrev powerExpr : Expr := .binOp (.name "d") .pow (.constant (.int 5))
private abbrev powerComp : Expr := .listComp powerExpr [("d", .call (.name "range") [.constant (.int 10)], [])]

private theorem power_loop (count first : Nat) (stop : Int) (length : (stop - first).toNat = count)
    (before : List Nat) (heap : Mutable.Heap) (env : HeapCalls.Env) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "d" first stop 1 [.comprehensionAppend heap.length powerExpr] []]
      (heap ++ [.list (encode before)]) finalEnv
      (heap ++ [.list (encode (before ++ (List.range' first count).map (· ^ 5)))]) := by
  induction count generalizing first before env with
  | zero =>
      refine ⟨env, ?_⟩
      simp only [List.range'_zero, List.map_nil, List.append_nil]
      apply BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let bound := setLocal env "d" (number first)
      obtain ⟨finalEnv, rest⟩ := ih (first + 1) (by omega) (before ++ [first ^ 5]) bound
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.append
        (body := [.comprehensionAppend heap.length powerExpr]) (middleEnv := bound)
        (middleHeap := heap ++ [.list (encode (before ++ [first ^ 5]))])
      · have powered : Evaluates program bound powerExpr (heap ++ [.list (encode before)])
            (number (first ^ 5)) (heap ++ [.list (encode before)]) := by
          unfold powerExpr
          refine Evaluates.binary (module := program) (env := bound) (globals := [])
            (a := .name "d") (b := .constant (.int 5)) (op := .pow)
            (heap := heap ++ [Mutable.Object.list (encode before)])
            (av := number first) (bv := .scalar (.int 5)) (value := number (first ^ 5))
            (middle := heap ++ [Mutable.Object.list (encode before)])
            (afterRight := heap ++ [Mutable.Object.list (encode before)])
            (finalHeap := heap ++ [Mutable.Object.list (encode before)]) ?_ ?_ ?_
          · exact Evaluates.lookup (by simp [bound, setLocal])
          · exact Evaluates.constant
          · simp [Mutable.binary, scalarBinary, number]
        have emit := BlockContinues.comprehension_append (address := heap.length)
          (items := encode before) powered (by simp)
        simpa [List.set_append_right, encode, List.map_append] using emit
      · simpa [List.range'_succ, List.append_assoc, Calls.loopContinuation, Calls.loopControl,
          Calls.loopControlStmt, Nat.cast_add, Nat.cast_one] using rest

private theorem make_powers (heap : Mutable.Heap) (env : HeapCalls.Env) :
    Evaluates program env powerComp heap (.ref heap.length) (heap ++ [.list powers]) := by
  obtain ⟨finalEnv, executed⟩ := power_loop 10 0 10 rfl [] heap
    (comprehensionEnv env [("d", .call (.name "range") [.constant (.int 10)], [])])
  apply Evaluates.list_comp_range (.cons Evaluates.constant .nil) rfl
  simpa [powers, List.range_eq_range'] using executed

private abbrev known := ProjectEulerStatements.FifthPowerDigits.solutions

private theorem finish (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (found : heap[address]? = some (.list (encode known)))
    (getMatches : env "matches" = some (.ref address)) :
    BlockReturns program env
      [.assign [.name "ans"] (.sumSequence (.name "matches") (.constant (.int 0))),
       .assert (.compare (.name "matches") [(.eq, .list
         [.constant (.int 4150), .constant (.int 4151), .constant (.int 54748),
          .constant (.int 92727), .constant (.int 93084), .constant (.int 194979)])]) none,
       .return (.name "ans")]
      heap (number 443839) (heap ++ [.list (encode known)]) := by
  apply BlockReturns.assign (value := number 443839) (middle := heap)
  · have summed := Evaluates.sum_integers (values := known.map Nat.cast)
      (initial := 0) (start := .constant (.int 0)) (globals := [])
      (module := program) (env := env) (heap := heap) (middle := heap)
      (.cons (Evaluates.lookup getMatches) (.cons Evaluates.constant .nil))
      (by simp [Mutable.sequence, found, encode, List.map_map])
    exact summed
  · apply BlockReturns.assertion
    · apply Evaluates.compare
        (Evaluates.lookup (by simpa [setLocal] using getMatches))
        (Evaluates.list (.cons Evaluates.constant (.cons Evaluates.constant
          (.cons Evaluates.constant (.cons Evaluates.constant
            (.cons Evaluates.constant (.cons Evaluates.constant .nil)))))))
      have compared := HeapCalls.compare_int_lists (heap := heap ++ [.list (encode known)])
        (a := address) (b := heap.length) (known.map Nat.cast) (known.map Nat.cast)
        (by simpa [encode, List.map_map] using found_append found [.list (encode known)])
        (by simp [encode, List.map_map])
      simpa using compared
    · rfl
    · exact BlockReturns.return_value (Evaluates.lookup (by simp [setLocal]))

/-- The literal result check in the Python source is proved as well as its return value. -/
theorem solve.result : ∃ finalHeap, ReturnsValue program "solve" [] [] (number 443839) finalHeap := by
  let initial : HeapCalls.Env := functionEnv
    ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [] []
  let fifthEnv := setLocal initial "fifth" (.ref 0)
  let upperEnv := setLocal fifthEnv "upper" (number 354294)
  let ready := setLocal upperEnv "matches" (.ref 1)
  obtain ⟨finalHeap, finalEnv, loop, found, bound⟩ := scan 354293 2 0 1 354295 rfl (by decide)
    [] [.list powers, .list []] ready (by simp) (by simp) rfl rfl
  have knownFound : finalHeap[1]? = some (.list (encode known)) := by
    simpa only [List.nil_append, ProjectEulerStatements.FifthPowerDigits.optimized_matches] using found
  refine ⟨finalHeap ++ [.list (encode known)], ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .ref 0) (middle := [.list powers])
    (make_powers [] initial)
  apply BlockReturns.assign (value := number 354294) (middle := [.list powers])
  · apply Evaluates.binary Evaluates.constant
      (Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup rfl) Evaluates.constant rfl) rfl
  · apply BlockContinues.returns (BlockContinues.annAssign
      (value := .ref 1) (finalHeap := [.list powers, .list []]) (Evaluates.list .nil))
    have scanned : BlockContinues program ready
        [.forIn (.name "n") (.call (.name "range")
          [.constant (.int 2), .binOp (.name "upper") .add (.constant (.int 1))]) scanBody []]
        [.list powers, .list []] finalEnv finalHeap := by
      apply BlockContinues.for_range
        (.cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl) .nil)) rfl
      exact loop
    exact scanned.returns (finish finalHeap finalEnv 1 knownFound bound)

theorem solve.«Equiv(naive_default)» :
    (HeapCalls.scalarFunction program "solve" []).Equiv
      (HeapCalls.scalarFunction P30Ast.module "naive_default" []) := by
  obtain ⟨_, returned⟩ := solve.result
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p30.naive_default.result []
  rw [ProjectEulerStatements.FifthPowerDigits.reference_value] at reference
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
      (HeapCalls.scalarFunction P30Ast.module "naive_default" []).Supported := by
  obtain ⟨_, returned⟩ := solve.result
  exact ⟨(HeapCalls.ReturnsValue.scalar_returns (values := []) returned).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p30.naive_default.«Supported»⟩

end solvers.«30»
