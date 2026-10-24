import ProjectEulerStatements.Lib.PythagoreanPerimeterCountingProof
import data.«project-euler-statements».data.py.project_euler_statements.«p39.py»
import «.formalsw».solvers.«39.py.ast»
import lang.py.FormalSW.HeapCalls
import Mathlib.Data.List.Range
import ProjectEulerStatements.Lib.ArgMaximum
import ProjectEulerStatements.Lib.PythagoreanPerimeterCounting

set_option Elab.async false

namespace solvers.«39»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P39Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

namespace solve

private abbrev code : List Stmt := executableBody
  ((findFunctionIn program.body "solve").getD ([], [])).2

/-- Counts remain natural numbers in the original mutable Python list. -/
private def table (size : Nat) (counts : Nat → Nat) : List Mutable.Value :=
  (List.range size).map (fun p => number (counts p))

@[simp] private theorem table_length (size : Nat) (counts : Nat → Nat) :
    (table size counts).length = size := by simp [table]

private theorem table_lookup (size : Nat) (counts : Nat → Nat) (index : Nat) (inside : index < size) :
    (table size counts)[index]? = some (number (counts index)) := by simp [table, inside]

private theorem table_set (size : Nat) (counts : Nat → Nat) (index value : Nat) (inside : index < size) :
    (table size counts).set index (number value) =
      table size (fun p => if p = index then value else counts p) := by
  apply List.ext_getElem?
  intro p
  by_cases bounded : p < size
  · by_cases same : index = p
    · subst p
      rw [List.getElem?_set_self (by simpa using inside), table_lookup size _ index inside]
      simp
    · rw [List.getElem?_set_ne same, table_lookup size counts p bounded, table_lookup size _ p bounded]
      simp [Ne.symm same]
  · have outside : size ≤ p := by omega
    simp only [List.getElem?_eq_none (by simpa using outside : ((table size counts).set index (number value)).length ≤ p),
      List.getElem?_eq_none (by simpa using outside : (table size (fun n => if n = index then value else counts n)).length ≤ p)]

private def arrayHeap (base : Mutable.Heap) (limit : Nat) (counts : Nat → Nat) : Mutable.Heap :=
  base ++ [.list (table (limit+1) counts)]

private theorem arrayHeap_store (base : Mutable.Heap) (limit : Nat) (old updated : Nat → Nat) :
    (arrayHeap base limit old).set base.length (.list (table (limit+1) updated)) =
      arrayHeap base limit updated := by simp [arrayHeap, List.set_append_right]

private theorem table_zero (size : Nat) : table size (fun _ => 0) = List.replicate size (number 0) := by
  simp [table]

private theorem initialize_result (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (number limit)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) heap (.ref (heap.length+1))
      (arrayHeap (heap ++ [.list [number 0]]) limit (fun _ => 0)) := by
  let base := heap ++ [.list [number 0]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number (limit+1))
    (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number])
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, base, arrayHeap, table_zero, Except.map]

private abbrev incrementBody : List Stmt :=
  [.augAssign (.subscript (.name "counts") (.binOp (.name "k") .mult (.name "p0"))) .add (.constant (.int 1))]

/-- The source indexed augmented assignment reads and writes the same list cell. -/
private theorem increment_source (limit k p0 address : Nat) (counts : Nat → Nat)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (inside : k*p0 ≤ limit)
    (getCounts : env "counts" = some (.ref address)) (getK : env "k" = some (number k))
    (getP : env "p0" = some (number p0))
    (found : heap[address]? = some (.list (table (limit+1) counts))) :
    BlockContinues program env incrementBody heap env
      (heap.set address (.list (table (limit+1) (fun p => if p = k*p0 then counts (k*p0)+1 else counts p)))) := by
  have index : Evaluates program env (.binOp (.name "k") .mult (.name "p0")) heap
      (number (k*p0)) heap := Evaluates.binary (Evaluates.lookup getK) (Evaluates.lookup getP)
        (by simp [Mutable.binary, scalarBinary, number])
  apply BlockContinues.augItem (by intro a b c; simp) (Evaluates.lookup getCounts) index
    (previous := number (counts (k*p0))) (updated := number (counts (k*p0)+1))
    (afterRead := heap) (afterOperation := heap) (rhs := Evaluates.constant)
  · have item := table_lookup (limit+1) counts (k*p0) (by omega)
    have bound : k*p0 < (table (limit+1) counts).length := by simp; omega
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using
      Mutable.getItem_list heap address (k*p0) _ found bound
  · simp [inplace, Mutable.binary, scalarBinary, number]
  · rw [← table_set (limit+1) counts (k*p0) (counts (k*p0)+1) (by omega)]
    have nonnegative : ¬ (k : Int)*p0 < 0 := by
      have product : (0 : Int) ≤ (k : Int)*p0 := Int.mul_nonneg (Int.natCast_nonneg k) (Int.natCast_nonneg p0)
      omega
    have bounded : ¬ (limit : Int)+1 ≤ (k : Int)*p0 := by
      have bound : ((k*p0 : Nat) : Int) ≤ limit := by exact_mod_cast inside
      simp only [Int.natCast_mul] at bound
      omega
    simp [Mutable.setItem, Mutable.index, found, nonnegative, bounded, number, Except.map]
    rw [← Int.natCast_mul, Int.toNat_natCast]

/- Append inside solve; requires Lib/ArgMaximum. -/
private abbrev selectionBody : List Stmt :=
  match (code[5]? : Option Stmt) with | some (.forIn _ _ statements _) => statements | _ => []

private theorem read_counts (limit p address : Nat) (counts : Nat → Nat)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (inside : p ≤ limit)
    (getCounts : env "counts" = some (.ref address)) (getP : env "p" = some (number p))
    (found : heap[address]? = some (.list (table (limit+1) counts))) :
    Evaluates program env (.subscript (.name "counts") (.name "p")) heap (number (counts p)) heap := by
  apply Evaluates.subscript (by intros; intro h; cases h) (Evaluates.lookup getCounts) (Evaluates.lookup getP)
  have item := table_lookup (limit+1) counts p (by omega)
  have bounded : p < (table (limit+1) counts).length := by simp; omega
  simpa only [(List.getElem?_eq_some_iff.mp item).2] using
    Mutable.getItem_list heap address p _ found bounded

private theorem selection_loop (limit first count best : Nat) (counts : Nat → Nat)
    (base : Mutable.Heap) (env : HeapCalls.Env) (endpoint : first+count=limit+1)
    (getCounts : env "counts" = some (.ref base.length))
    (getBest : env "best_p" = some (number best)) (getScore : env "best_count" = some (number (counts best))) :
    BlockReturns program env [.rangeStepLoop "p" first (limit+1) 1 selectionBody [], .return (.name "best_p")]
      (arrayHeap base limit counts)
      (number ((List.range' first count).foldl (ProjectEulerStatements.ArgMaximum.select counts) best))
      (arrayHeap base limit counts) := by
  induction count generalizing first best env with
  | zero =>
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup getBest)
  | succ count ih =>
      let bound := setLocal env "p" (number first)
      have boundCounts : bound "counts" = some (.ref base.length) := by simpa [bound, setLocal] using getCounts
      have boundBest : bound "best_p" = some (number best) := by simpa [bound, setLocal] using getBest
      have boundScore : bound "best_count" = some (number (counts best)) := by simpa [bound, setLocal] using getScore
      have read := read_counts limit first base.length counts bound (arrayHeap base limit counts)
        (by omega) boundCounts (by simp [bound, setLocal]) (by simp [arrayHeap])
      have tested : Evaluates program bound
          (.compare (.subscript (.name "counts") (.name "p")) [(.gt, .name "best_count")])
          (arrayHeap base limit counts) (.scalar (.bool (decide (counts best < counts first)))) (arrayHeap base limit counts) :=
        Evaluates.compare read (Evaluates.lookup boundScore) (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
      simp only [List.range'_succ, List.foldl_cons]
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      by_cases better : counts best < counts first
      · simp only [ProjectEulerStatements.ArgMaximum.select, if_pos better]
        let e1 := setLocal bound "best_count" (number (counts first))
        let updated := setLocal e1 "best_p" (number first)
        have rest := ih (first+1) first updated (by omega)
          (by simpa [updated, e1, setLocal] using boundCounts)
          (by simp [updated, setLocal]) (by simp [updated, e1, setLocal])
        apply BlockReturns.branch (flag := true) (by simpa [better] using tested) rfl
        apply BlockReturns.assign read
        apply BlockReturns.assign (Evaluates.lookup (value := number first) (by simp [bound, setLocal]))
        simpa [updated, e1, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest
      · simp only [ProjectEulerStatements.ArgMaximum.select, if_neg better]
        have rest := ih (first+1) best bound (by omega) boundCounts boundBest boundScore
        apply BlockReturns.branch (flag := false) (by simpa [better] using tested) rfl
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

private theorem selection_result (limit : Nat) (counts : Nat → Nat) (zero : counts 0 = 0)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) (getCounts : env "counts" = some (.ref base.length)) :
    BlockReturns program env (code.drop 3) (arrayHeap base limit counts)
      (number ((List.range' 1 limit).foldl (ProjectEulerStatements.ArgMaximum.select counts) 0))
      (arrayHeap base limit counts) := by
  let e1 := setLocal env "best_p" (number 0)
  let ready := setLocal e1 "best_count" (number 0)
  have loop := selection_loop limit 1 limit 0 counts base ready (by omega)
    (by simpa [ready, e1, setLocal] using getCounts) (by simp [ready, e1, setLocal])
    (by simp [ready, setLocal, zero])
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range (values := [number 1, number (limit+1)])
    (.cons Evaluates.constant (.cons
      (Evaluates.binary (av := number limit) (bv := number 1) (value := number (limit+1))
        (middle := arrayHeap base limit counts) (afterRight := arrayHeap base limit counts)
        (finalHeap := arrayHeap base limit counts)
        (Evaluates.lookup (by simpa [ready, e1, setLocal] using getLimit)) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
  exact loop

/- Append inside solve; imports PythagoreanPerimeterCounting. -/
open ProjectEulerStatements.PythagoreanPerimeterCounting
  (perimeter permitted addMultiples addMultiples_zero addMultiples_step
    row row_zero row_step outer outer_zero outer_step outer_inactive)
private abbrev bump := ProjectEulerStatements.PythagoreanPerimeterCounting.increment
private abbrev whileCondition : Expr := .compare
  (.binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult
    (.binOp (.name "m") .add (.constant (.int 1)))) [(.ltE, .name "limit")]
private abbrev whileBody : List Stmt :=
  match (code[2]? : Option Stmt) with | some (.whileDo _ statements _) => statements | _ => []
private abbrev rowBody : List Stmt :=
  match (whileBody[0]? : Option Stmt) with | some (.forIn _ _ statements _) => statements | _ => []

private def Iteration (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap) : Prop :=
  ∀ next following anchors stack, Reaches program anchors
    ⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩
    ⟨.block (next :: following) finalEnv, stack, finalHeap⟩

private theorem Iteration.done (env : HeapCalls.Env) (heap : Mutable.Heap) : Iteration env [] heap env heap := by
  intro next following anchors stack
  exact .next rfl .refl

private theorem Iteration.prepend
    (first : BlockContinues program env body heap middleEnv middleHeap)
    (rest : Iteration middleEnv tail middleHeap finalEnv finalHeap) :
    Iteration env (body ++ tail) heap finalEnv finalHeap := by
  intro next following anchors stack
  have tailRun : Reaches program anchors
      ⟨.block (tail ++ (.loopResume next :: following)) middleEnv, stack, middleHeap⟩
      ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ := by
    simpa only [List.append_assoc, List.singleton_append] using rest next following anchors stack
  simpa only [List.append_assoc, List.singleton_append] using
    (first (tail ++ (.loopResume next :: following)) anchors stack).trans tailRun

private theorem Iteration.branch
    (evaluated : Evaluates program env condition heap value middle)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : Iteration env ((if flag then yes else no) ++ body) middle finalEnv finalHeap) :
    Iteration env (.ifThen condition yes no :: body) heap finalEnv finalHeap := by
  intro next following anchors stack
  apply Reaches.next rfl
  apply (evaluated anchors _).trans
  apply Reaches.next (following :=
    ⟨.block (((if flag then yes else no) ++ body) ++ [.loopResume next] ++ following) env, stack, middle⟩)
    (by simp [HeapCalls.step, run, truth, List.append_assoc])
  exact rest next following anchors stack

private theorem Iteration.skip
    (target : ∀ next following, Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following)) :
    Iteration env (.continueLoop :: body) heap env heap := by
  intro next following anchors stack
  have found : Calls.loopTarget (body ++ (.loopResume next :: following)) = some (next, following) := by
    simpa only [List.append_assoc, List.singleton_append] using target next following
  exact .next (by simp [HeapCalls.step, found]) .refl

private structure Locals (env : HeapCalls.Env) (limit address m : Nat) : Prop where
  bound : env "limit" = some (number limit)
  array : env "counts" = some (.ref address)
  outer : env "m" = some (number m)

private theorem multiple_loop (limit m n first count : Nat) (counts : Nat → Nat)
    (positive : 0 < perimeter m n) (endpoint : first+count=limit/perimeter m n+1)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (getArray : env "counts" = some (.ref base.length)) (getP : env "p0" = some (number (perimeter m n))) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "k" first (limit/perimeter m n+1) 1 incrementBody []]
      (arrayHeap base limit counts) finalEnv (arrayHeap base limit (addMultiples limit m n first count counts)) ∧
      ∀ query, query ≠ "k" → finalEnv query = env query := by
  induction count generalizing first counts env with
  | zero =>
      refine ⟨env, ?_, fun _ _ => rfl⟩
      rw [addMultiples_zero]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "k" (number first)
      have boundArray : bound "counts" = some (.ref base.length) := by simpa [bound, setLocal] using getArray
      have boundP : bound "p0" = some (number (perimeter m n)) := by simpa [bound, setLocal] using getP
      have inside : first * perimeter m n ≤ limit := (Nat.le_div_iff_mul_le positive).mp (by omega)
      have written := increment_source limit first (perimeter m n) base.length counts bound
        (arrayHeap base limit counts) inside boundArray (by simp [bound, setLocal]) boundP (by simp [arrayHeap])
      have same : (fun p => if p = first*perimeter m n then counts (first*perimeter m n)+1 else counts p) =
          bump counts (first*perimeter m n) := by
        funext p
        by_cases equal : p = first*perimeter m n <;> simp [bump, ProjectEulerStatements.PythagoreanPerimeterCounting.increment, equal]
      rw [same, arrayHeap_store] at written
      obtain ⟨finalEnv, rest, stable⟩ := ih (first+1) (bump counts (first*perimeter m n)) (by omega)
        bound boundArray boundP
      refine ⟨finalEnv, ?_, ?_⟩
      · rw [addMultiples_step]
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using written.append rest
      · intro query different
        exact (stable query different).trans (by simp [bound, setLocal, different])

private theorem parity_test (m n : Nat) (inside : n ≤ m) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getM : env "m" = some (number m)) (getN : env "n" = some (number n)) :
    Evaluates program env
      (.compare (.binOp (.binOp (.name "m") .sub (.name "n")) .bitAnd (.constant (.int 1)))
        [(.eq, .constant (.int 0))]) heap (.scalar (.bool (decide ((m-n)%2=0)))) heap := by
  have difference : Evaluates program env (.binOp (.name "m") .sub (.name "n")) heap (number (m-n)) heap :=
    Evaluates.binary (Evaluates.lookup getM) (Evaluates.lookup getN)
      (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub inside])
  have parity : Evaluates program env (.binOp (.binOp (.name "m") .sub (.name "n")) .bitAnd (.constant (.int 1)))
      heap (number ((m-n)%2)) heap :=
    Evaluates.binary difference Evaluates.constant (by simp [Mutable.binary, scalarBitAnd, number])
  exact Evaluates.compare parity Evaluates.constant
    (by
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      norm_cast
      exact Nat.dvd_iff_mod_eq_zero)

private theorem gcd_test (m n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getM : env "m" = some (number m)) (getN : env "n" = some (number n)) :
    Evaluates program env (.compare (.call (.name "gcd") [.name "m", .name "n"])
      [(.notEq, .constant (.int 1))]) heap (.scalar (.bool (decide (Nat.gcd m n ≠ 1)))) heap := by
  have divisor : Evaluates program env (.call (.name "gcd") [.name "m", .name "n"]) heap (number (Nat.gcd m n)) heap := by
    apply Evaluates.library_call (.cons (Evaluates.lookup getM) (.cons (Evaluates.lookup getN) .nil)) rfl
    change scalarResult (some (Math.gcdCall [.int m, .int n])) heap = _
    rw [Math.gcdCall_pair]
    simp [scalarResult, number]
  exact Evaluates.compare divisor Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

private theorem perimeter_value (m n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getM : env "m" = some (number m)) (getN : env "n" = some (number n)) :
    Evaluates program env (.binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult
      (.binOp (.name "m") .add (.name "n"))) heap (number (perimeter m n)) heap := by
  apply Evaluates.binary (av := number (2*m)) (bv := number (m+n)) (middle := heap) (afterRight := heap)
  · exact Evaluates.binary Evaluates.constant (Evaluates.lookup getM) (by simp [Mutable.binary, scalarBinary, number])
  · exact Evaluates.binary (Evaluates.lookup getM) (Evaluates.lookup getN) (by simp [Mutable.binary, scalarBinary, number])
  · simp [Mutable.binary, scalarBinary, number, perimeter]

private theorem multiple_for (limit m n : Nat) (counts : Nat → Nat) (positive : 0 < perimeter m n)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (getArray : env "counts" = some (.ref base.length)) (getP : env "p0" = some (number (perimeter m n)))
    (getLimit : env "limit" = some (number limit)) :
    ∃ finalEnv, BlockContinues program env [rowBody[4]'(by decide)] (arrayHeap base limit counts)
      finalEnv (arrayHeap base limit (addMultiples limit m n 1 (limit/perimeter m n) counts)) ∧
      ∀ query, query ≠ "k" → finalEnv query = env query := by
  obtain ⟨finalEnv, executed, stable⟩ := multiple_loop limit m n 1 (limit/perimeter m n) counts positive
    (by omega) base env getArray getP
  refine ⟨finalEnv, ?_, stable⟩
  apply BlockContinues.for_range (values := [number 1, number (limit/perimeter m n+1)])
    (.cons Evaluates.constant (.cons
      (Evaluates.binary (av := number (limit/perimeter m n)) (bv := number 1)
        (value := number (limit/perimeter m n+1)) (middle := arrayHeap base limit counts)
        (afterRight := arrayHeap base limit counts) (finalHeap := arrayHeap base limit counts)
        (Evaluates.binary (av := number limit) (bv := number (perimeter m n))
          (value := number (limit/perimeter m n)) (middle := arrayHeap base limit counts)
          (afterRight := arrayHeap base limit counts) (finalHeap := arrayHeap base limit counts)
          (Evaluates.lookup getLimit) (Evaluates.lookup getP)
          (by simp [Mutable.binary, scalarBinary, number, positive, show perimeter m n ≠ 0 by omega]))
        Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
  exact executed

/- Append inside solve after the multiples helpers. -/
private theorem row_iteration (limit m n : Nat) (counts : Nat → Nat)
    (lower : 1 ≤ n) (upper : n < m) (base : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals env limit base.length m) (getN : env "n" = some (number n)) :
    ∃ finalEnv, Iteration env rowBody (arrayHeap base limit counts) finalEnv
      (arrayHeap base limit (if permitted limit m n then
        addMultiples limit m n 1 (limit/perimeter m n) counts else counts)) ∧
      Locals finalEnv limit base.length m := by
  have parity := parity_test m n (by omega) env (arrayHeap base limit counts) locals.outer getN
  by_cases even : (m-n)%2=0
  · have rejected : permitted limit m n = false := by simp [permitted, even]
    simp only [rejected, Bool.false_eq_true, if_false]
    refine ⟨env, ?_, locals⟩
    apply Iteration.branch (flag := true) (by simpa [even] using parity) rfl
    apply Iteration.skip
    intro next following
    rfl
  · have odd : (m-n)%2=1 := by omega
    have divisor := gcd_test m n env (arrayHeap base limit counts) locals.outer getN
    by_cases bad : Nat.gcd m n ≠ 1
    · have rejected : permitted limit m n = false := by simp [permitted, Nat.Coprime, bad]
      simp only [rejected, Bool.false_eq_true, if_false]
      refine ⟨env, ?_, locals⟩
      apply Iteration.branch (flag := false) (by simpa [even] using parity) rfl
      apply Iteration.branch (flag := true) (by simpa [bad] using divisor) rfl
      apply Iteration.skip
      intro next following
      rfl
    · have coprime : Nat.gcd m n = 1 := by omega
      let ready := setLocal env "p0" (number (perimeter m n))
      have readyLocals : Locals ready limit base.length m := by
        exact ⟨by simpa [ready, setLocal] using locals.bound,
          by simpa [ready, setLocal] using locals.array, by simpa [ready, setLocal] using locals.outer⟩
      have allocated : BlockContinues program env [rowBody[2]'(by decide)] (arrayHeap base limit counts)
          ready (arrayHeap base limit counts) :=
        BlockContinues.assign (perimeter_value m n env (arrayHeap base limit counts) locals.outer getN)
      have tested : Evaluates program ready (.compare (.name "p0") [(.gt, .name "limit")])
          (arrayHeap base limit counts) (.scalar (.bool (decide (limit < perimeter m n))))
          (arrayHeap base limit counts) := Evaluates.compare (Evaluates.lookup rfl)
            (Evaluates.lookup readyLocals.bound) (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
      by_cases large : limit < perimeter m n
      · have rejected : permitted limit m n = false := by simp [permitted, Nat.not_le.mpr large]
        simp only [rejected, Bool.false_eq_true, if_false]
        refine ⟨ready, ?_, readyLocals⟩
        apply Iteration.branch (flag := false) (by simpa [even] using parity) rfl
        apply Iteration.branch (flag := false) (by simpa [bad] using divisor) rfl
        apply Iteration.prepend allocated
        apply Iteration.branch (flag := true) (by simpa [large] using tested) rfl
        apply Iteration.skip
        intro next following
        rfl
      · have permittedYes : permitted limit m n = true := by simp [permitted, odd, Nat.Coprime, coprime, Nat.le_of_not_gt large]
        simp only [permittedYes, if_true]
        have positive : 0 < perimeter m n := by
          exact Nat.mul_pos (Nat.mul_pos (by decide) (by omega)) (by omega)
        obtain ⟨finalEnv, executed, saved⟩ := multiple_for limit m n counts positive base ready
          readyLocals.array rfl readyLocals.bound
        refine ⟨finalEnv, ?_, ?_⟩
        · apply Iteration.branch (flag := false) (by simpa [even] using parity) rfl
          apply Iteration.branch (flag := false) (by simpa [bad] using divisor) rfl
          apply Iteration.prepend allocated
          apply Iteration.branch (flag := false) (by simpa [large] using tested) rfl
          simpa only [Bool.false_eq_true, if_false, List.nil_append, List.append_nil] using
            Iteration.prepend executed (Iteration.done _ _)
        · exact ⟨(saved "limit" (by decide)).trans readyLocals.bound,
            (saved "counts" (by decide)).trans readyLocals.array,
            (saved "m" (by decide)).trans readyLocals.outer⟩

private theorem row_loop (limit m first count : Nat) (counts : Nat → Nat)
    (lower : 1 ≤ first) (endpoint : first+count=m)
    (base : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env limit base.length m) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "n" first m 1 rowBody []]
      (arrayHeap base limit counts) finalEnv (arrayHeap base limit (row limit m first count counts)) ∧
      Locals finalEnv limit base.length m := by
  induction count generalizing first counts env with
  | zero =>
      refine ⟨env, ?_, locals⟩
      rw [row_zero]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "n" (number first)
      have boundLocals : Locals bound limit base.length m :=
        ⟨by simpa [bound, setLocal] using locals.bound, by simpa [bound, setLocal] using locals.array,
          by simpa [bound, setLocal] using locals.outer⟩
      obtain ⟨middleEnv, iterated, middleLocals⟩ := row_iteration limit m first counts lower (by omega)
        base bound boundLocals (by simp [bound, setLocal])
      let nextCounts := if permitted limit m first then addMultiples limit m first 1 (limit/perimeter m first) counts else counts
      obtain ⟨finalEnv, rest, finalLocals⟩ := ih (first+1) nextCounts (by omega) (by omega) middleEnv middleLocals
      refine ⟨finalEnv, ?_, finalLocals⟩
      rw [row_step]
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      intro following anchors stack
      have path := iterated (.rangeStepLoop "n" ((first : Int)+1) m 1 rowBody []) following anchors stack
      have tailRun := rest following anchors stack
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, List.append_assoc, nextCounts]
        using path.trans tailRun

private theorem while_test (limit m : Nat) (base : Mutable.Heap) (counts : Nat → Nat)
    (env : HeapCalls.Env) (locals : Locals env limit base.length m) :
    Evaluates program env whileCondition (arrayHeap base limit counts)
      (.scalar (.bool (decide (2*m*(m+1)≤limit)))) (arrayHeap base limit counts) := by
  have left : Evaluates program env (.binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult
      (.binOp (.name "m") .add (.constant (.int 1)))) (arrayHeap base limit counts)
      (number (2*m*(m+1))) (arrayHeap base limit counts) := by
    apply Evaluates.binary (av := number (2*m)) (bv := number (m+1))
      (middle := arrayHeap base limit counts) (afterRight := arrayHeap base limit counts)
    · exact Evaluates.binary Evaluates.constant (Evaluates.lookup locals.outer) (by simp [Mutable.binary, scalarBinary, number])
    · exact Evaluates.binary (Evaluates.lookup locals.outer) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
    · simp [Mutable.binary, scalarBinary, number]
  exact Evaluates.compare left (Evaluates.lookup locals.bound)
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; norm_cast)

private theorem outer_loop (limit m count : Nat) (counts : Nat → Nat) (lower : 2 ≤ m)
    (endBound : limit < 2*(m+count)*(m+count+1))
    (base : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env limit base.length m) :
    ∃ finalEnv, BlockContinues program env [.whileDo whileCondition whileBody []]
      (arrayHeap base limit counts) finalEnv (arrayHeap base limit (outer limit m count counts)) ∧
      finalEnv "limit" = some (number limit) ∧ finalEnv "counts" = some (.ref base.length) := by
  induction count generalizing m counts env with
  | zero =>
      have stopped : ¬ 2*m*(m+1)≤limit := by simpa using Nat.not_le.mpr endBound
      have tested := while_test limit m base counts env locals
      refine ⟨env, ?_, locals.bound, locals.array⟩
      rw [outer_zero]
      exact BlockContinues.while_loop (flag := false) (by simpa [stopped] using tested) rfl .nil
  | succ count ih =>
      have tested := while_test limit m base counts env locals
      by_cases active : 2*m*(m+1)≤limit
      · obtain ⟨middleEnv, iterated, middleLocals⟩ := row_loop limit m 1 (m-1) counts (by decide) (by omega)
          base env locals
        have rowRun : BlockContinues program env [whileBody[0]'(by decide)] (arrayHeap base limit counts)
            middleEnv (arrayHeap base limit (row limit m 1 (m-1) counts)) := by
          apply BlockContinues.for_range (values := [number 1, number m])
            (.cons Evaluates.constant (.cons (Evaluates.lookup locals.outer) .nil)) rfl
          exact iterated
        let ready := setLocal middleEnv "m" (number (m+1))
        have incremented : BlockContinues program middleEnv [whileBody[1]'(by decide)]
            (arrayHeap base limit (row limit m 1 (m-1) counts)) ready (arrayHeap base limit (row limit m 1 (m-1) counts)) :=
          BlockContinues.augAssign middleLocals.outer Evaluates.constant (by simp [inplace, Mutable.binary, scalarBinary, number])
        have readyLocals : Locals ready limit base.length (m+1) :=
          ⟨by simpa [ready, setLocal] using middleLocals.bound, by simpa [ready, setLocal] using middleLocals.array,
            by simp [ready, setLocal]⟩
        obtain ⟨finalEnv, rest, finalLimit, finalArray⟩ := ih (m+1) (row limit m 1 (m-1) counts)
          (by omega) (by
            rw [show (m+1)+count = m+(count+1) by omega]
            exact endBound) ready readyLocals
        refine ⟨finalEnv, ?_, finalLimit, finalArray⟩
        rw [outer_step, if_pos active]
        apply BlockContinues.while_loop (flag := true) (by simpa [active] using tested) rfl
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using
          (rowRun.append incremented).append rest
      · refine ⟨env, ?_, locals.bound, locals.array⟩
        rw [outer_inactive limit m (count+1) counts (by omega)]
        exact BlockContinues.while_loop (flag := false) (by simpa [active] using tested) rfl .nil

private abbrev Counting := ProjectEulerStatements.PythagoreanPerimeterCounting.table

/-- Execute the actual mutable counter table, terminating while loop, and strict argmax. -/
theorem result (limit : Nat) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number limit] heap
      (number (ProjectEulerStatements.P39.naive limit)) finalHeap := by
  let definition := (findFunctionIn program.body "solve").getD ([], [])
  let initial := functionEnv definition.1 definition.2 [number limit] []
  let base := heap ++ [.list [number 0]]
  let allocated := setLocal initial "counts" (.ref base.length)
  let ready := setLocal allocated "m" (number 2)
  have initialLimit : initial "limit" = some (number limit) := rfl
  have readyLocals : Locals ready limit base.length 2 :=
    ⟨by simp [ready, allocated, setLocal, initialLimit],
      by simp [ready, allocated, setLocal], by simp [ready, setLocal]⟩
  have endBound : limit < 2 * (2 + (limit - 1)) * (2 + (limit - 1) + 1) := by
    have beyond : limit < 2 + (limit - 1) := by omega
    nlinarith
  obtain ⟨finalEnv, built, finalLimit, finalCounts⟩ :=
    outer_loop limit 2 (limit - 1) (fun _ => 0) (by decide) endBound base ready readyLocals
  have selected := selection_result limit (Counting limit)
    (ProjectEulerStatements.PythagoreanPerimeterCounting.table_zero limit)
    base finalEnv finalLimit finalCounts
  rw [ProjectEulerStatements.PythagoreanPerimeterCounting.table_scan] at selected
  refine ⟨arrayHeap base limit (Counting limit), ?_⟩
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  apply BlockContinues.returns (BlockContinues.annAssign (value := .ref base.length)
    (finalHeap := arrayHeap base limit (fun _ => 0)) ?_)
  · apply BlockReturns.assign Evaluates.constant
    exact built.returns selected
  · simpa only [base, List.length_append, List.length_singleton] using
      initialize_result limit initial heap initialLimit

private abbrev inputEnv (limit : Int) : ScalarEnv :=
  fun name => if name = "limit" then some (.int limit) else none

private theorem accepted_limit (limit : Int)
    (requires : Calls.Condition program (inputEnv limit)
      Solvers.P39Ast.requires_solve_0) : 0 ≤ limit := by
  simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare requires rfl

theorem «Equiv(naive)» (limit : Int)
    (requires : Calls.Condition program (inputEnv limit)
      Solvers.P39Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (Calls.function P39Ast.module "naive" [.int limit]) := by
  have nonnegative := accepted_limit limit requires
  lift limit to Nat using nonnegative
  obtain ⟨_, returned⟩ := result limit []
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := [.int limit]) returned
  have reference :=
    (data.«project-euler-statements».data.py.project_euler_statements.p39.naive.result limit).finishes
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ limit : Int, Calls.Condition program (inputEnv limit)
      Solvers.P39Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, Calls.Condition program (inputEnv limit)
      Solvers.P39Ast.requires_solve_0 →
      (Calls.function P39Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit requires
    have nonnegative := accepted_limit limit requires
    lift limit to Nat using nonnegative
    obtain ⟨_, returned⟩ := result limit []
    exact (HeapCalls.ReturnsValue.scalar_returns (values := [.int limit]) returned).supported
  · intro limit _
    exact data.«project-euler-statements».data.py.project_euler_statements.p39.naive.«Supported» limit

end solve
end solvers.«39»
