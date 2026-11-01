import ProjectEulerStatements.Lib.PythagoreanPerimeterBounds
import «.formalsw».solvers.«75.py.ast»
import lang.py.FormalSW.LoopCalls
import lang.py.FormalSW.SumCalls

/-! Execute the mutable perimeter sieve with its actual range cursors and loop transfers. -/
namespace solvers.«75»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
abbrev program := Solvers.P75Ast.module
abbrev integer (n : Int) : Mutable.Value := .scalar (.int n)
abbrev number (n : Nat) : Mutable.Value := integer n

namespace solve
open ProjectEulerStatements.PythagoreanPerimeterCounting
  (perimeter permitted addMultiples addMultiples_zero addMultiples_step
    row row_zero row_step row_inactive outer outer_zero outer_step maxParameter)
set_option Elab.async false

abbrev code : List Stmt := executableBody
  ((findFunctionIn program.body "solve").getD ([], [])).2

/-- Counts remain natural numbers in the original mutable Python list. -/
def table (size : Nat) (counts : Nat → Nat) : List Mutable.Value :=
  (List.range size).map (fun p => number (counts p))

@[simp] theorem table_length (size : Nat) (counts : Nat → Nat) :
    (table size counts).length = size := by simp [table]

theorem table_lookup (size : Nat) (counts : Nat → Nat) (index : Nat) (inside : index < size) :
    (table size counts)[index]? = some (number (counts index)) := by simp [table, inside]

theorem table_set (size : Nat) (counts : Nat → Nat) (index value : Nat) (inside : index < size) :
    (table size counts).set index (number value) =
      table size (fun p => if p = index then value else counts p) := by
  apply List.ext_getElem?
  intro p
  by_cases bounded : p < size
  · by_cases same : index = p
    · subst p
      rw [List.getElem?_set_self (by simpa using inside), table_lookup size _ index inside]
      simp
    · rw [List.getElem?_set_ne same, table_lookup size counts p bounded,
        table_lookup size _ p bounded]
      simp [Ne.symm same]
  · have outside : size ≤ p := by omega
    simp only [List.getElem?_eq_none (by simpa using outside :
      ((table size counts).set index (number value)).length ≤ p),
      List.getElem?_eq_none (by simpa using outside :
        (table size (fun n => if n = index then value else counts n)).length ≤ p)]

def arrayHeap (base : Mutable.Heap) (limit : Nat) (counts : Nat → Nat) : Mutable.Heap :=
  base ++ [.list (table (limit+1) counts)]

theorem arrayHeap_store (base : Mutable.Heap) (limit : Nat) (old updated : Nat → Nat) :
    (arrayHeap base limit old).set base.length (.list (table (limit+1) updated)) =
      arrayHeap base limit updated := by simp [arrayHeap, List.set_append_right]

theorem table_zero (size : Nat) : table size (fun _ => 0) = List.replicate size (number 0) := by
  simp [table]

theorem initialize_result (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (number limit)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) heap (.ref (heap.length+1))
      (arrayHeap (heap ++ [.list [number 0]]) limit (fun _ => 0)) := by
  let base := heap ++ [.list [number 0]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number (limit+1))
    (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number, integer])
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, integer, base, arrayHeap, table_zero, Except.map]

theorem gcd_test (m n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getM : env "m" = some (number m)) (getN : env "n" = some (number n)) :
    Evaluates program env (.compare (.call (.name "gcd") [.name "m", .name "n"])
      [(.notEq, .constant (.int 1))]) heap (.scalar (.bool (decide (Nat.gcd m n ≠ 1)))) heap := by
  have divisor : Evaluates program env (.call (.name "gcd") [.name "m", .name "n"])
      heap (number (Nat.gcd m n)) heap := by
    apply Evaluates.library_call
      (.cons (Evaluates.lookup getM) (.cons (Evaluates.lookup getN) .nil)) rfl
    change scalarResult (some (Math.gcdCall [.int m, .int n])) heap = _
    rw [Math.gcdCall_pair]
    simp [scalarResult, number]
  exact Evaluates.compare divisor Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

theorem perimeter_value (m n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getM : env "m" = some (number m)) (getN : env "n" = some (number n)) :
    Evaluates program env (.binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult
      (.binOp (.name "m") .add (.name "n"))) heap (number (perimeter m n)) heap := by
  apply Evaluates.binary (av := number (2*m)) (bv := number (m+n))
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary Evaluates.constant (Evaluates.lookup getM)
      (by simp [Mutable.binary, scalarBinary, number, integer])
  · exact Evaluates.binary (Evaluates.lookup getM) (Evaluates.lookup getN)
      (by simp [Mutable.binary, scalarBinary, number, integer])
  · simp [Mutable.binary, scalarBinary, number, perimeter]


abbrev outerBody : List Stmt :=
  match (code[3]? : Option Stmt) with | some (.forIn _ _ statements _) => statements | _ => []
abbrev rowBody : List Stmt :=
  match (outerBody[0]? : Option Stmt) with | some (.forIn _ _ statements _) => statements | _ => []
abbrev incrementBody : List Stmt :=
  [.augAssign (.subscript (.name "counts") (.name "p")) .add (.constant (.int 1))]

/-- The indexed augmented assignment changes exactly the in-range perimeter cell. -/
theorem increment_source (limit p address : Nat) (counts : Nat → Nat)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (inside : p ≤ limit)
    (getCounts : env "counts" = some (.ref address)) (getP : env "p" = some (number p))
    (found : heap[address]? = some (.list (table (limit+1) counts))) :
    BlockContinues program env incrementBody heap env
      (heap.set address (.list (table (limit+1)
        (ProjectEulerStatements.PythagoreanPerimeterCounting.increment counts p)))) := by
  apply BlockContinues.augItem (by intro a b c; simp) (Evaluates.lookup getCounts)
    (Evaluates.lookup getP) (previous := number (counts p)) (updated := number (counts p+1))
    (afterRead := heap) (afterOperation := heap) (rhs := Evaluates.constant)
  · have item := table_lookup (limit+1) counts p (by omega)
    have bound : p < (table (limit+1) counts).length := by simp; omega
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using
      Mutable.getItem_list heap address p _ found bound
  · simp [inplace, Mutable.binary, scalarBinary, number, integer]
  · have updated : ProjectEulerStatements.PythagoreanPerimeterCounting.increment counts p =
        (fun q => if q = p then counts p+1 else counts q) := by
      funext q
      by_cases same : q = p <;>
        simp [ProjectEulerStatements.PythagoreanPerimeterCounting.increment, same]
    rw [updated, ← table_set (limit+1) counts p (counts p+1) (by omega)]
    simp [Mutable.setItem, Mutable.index, found, number, integer,
      show ¬ (p : Int) < 0 by omega, not_lt.mpr inside, Except.map]

theorem multiple_loop (limit m n first count : Nat) (counts : Nat → Nat)
    (positive : 0 < perimeter m n) (endpoint : first+count=limit/perimeter m n+1)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (getArray : env "counts" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "p" (first * perimeter m n) (limit+1) (perimeter m n) incrementBody []]
      (arrayHeap base limit counts) finalEnv
      (arrayHeap base limit (addMultiples limit m n first count counts)) ∧
      ∀ query, query ≠ "p" → finalEnv query = env query := by
  induction count generalizing first counts env with
  | zero =>
      have outside : limit < first * perimeter m n := by
        have bound := Nat.lt_mul_div_succ limit positive
        rw [Nat.mul_comm] at bound
        have equal : first = limit/perimeter m n+1 := by omega
        simpa only [equal] using bound
      refine ⟨env, ?_, fun _ _ => rfl⟩
      rw [addMultiples_zero]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues, positive]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "p" (number (first * perimeter m n))
      have boundArray : bound "counts" = some (.ref base.length) := by
        simpa [bound, setLocal] using getArray
      have inside : first * perimeter m n ≤ limit :=
        (Nat.le_div_iff_mul_le positive).mp (by omega)
      have written := increment_source limit (first * perimeter m n) base.length counts bound
        (arrayHeap base limit counts) inside boundArray (by simp [bound, setLocal])
        (by simp [arrayHeap])
      rw [arrayHeap_store] at written
      obtain ⟨finalEnv, rest, stable⟩ := ih (first+1)
        (ProjectEulerStatements.PythagoreanPerimeterCounting.increment counts
          (first * perimeter m n)) (by omega) bound boundArray
      refine ⟨finalEnv, ?_, ?_⟩
      · rw [addMultiples_step]
        apply BlockContinues.range_step (by simp [Calls.rangeContinues, positive]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Nat.add_mul, Nat.cast_add, Nat.cast_mul, Int.add_mul, bound, number, integer]
          using written.append rest
      · intro query different
        exact (stable query different).trans (by simp [bound, setLocal, different])

theorem multiple_for (limit m n : Nat) (counts : Nat → Nat) (positive : 0 < perimeter m n)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (getArray : env "counts" = some (.ref base.length))
    (getP : env "p0" = some (number (perimeter m n)))
    (getLimit : env "limit" = some (number limit)) :
    ∃ finalEnv, BlockContinues program env [rowBody[4]'(by decide)]
      (arrayHeap base limit counts) finalEnv
      (arrayHeap base limit (addMultiples limit m n 1 (limit/perimeter m n) counts)) ∧
      ∀ query, query ≠ "p" → finalEnv query = env query := by
  obtain ⟨finalEnv, executed, stable⟩ := multiple_loop limit m n 1 (limit/perimeter m n)
    counts positive (by omega) base env getArray
  refine ⟨finalEnv, ?_, stable⟩
  apply BlockContinues.for_range (start := perimeter m n) (stop := limit+1)
    (stride := perimeter m n)
    (values := [number (perimeter m n), number (limit+1), number (perimeter m n)])
    (.cons (Evaluates.lookup getP) (.cons
      (Evaluates.binary (av := number limit) (bv := number 1)
        (value := number (limit+1)) (middle := arrayHeap base limit counts)
        (afterRight := arrayHeap base limit counts) (finalHeap := arrayHeap base limit counts)
        (Evaluates.lookup getLimit) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number, integer]))
      (.cons (Evaluates.lookup getP) .nil)))
    (by
      change (if (perimeter m n : Int) = 0 then _ else _) = _
      rw [if_neg (by omega : (perimeter m n : Int) ≠ 0)]
      rfl)
  simpa only [Nat.cast_one, one_mul] using executed

theorem parity_test (m n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getM : env "m" = some (number m)) (getN : env "n" = some (number n)) :
    Evaluates program env
      (.compare (.binOp (.binOp (.name "m") .bitXor (.name "n")) .bitAnd (.constant (.int 1)))
        [(.eq, .constant (.int 0))]) heap
      (.scalar (.bool (decide (((m ^^^ n) &&& 1) = 0)))) heap := by
  have xor : Evaluates program env (.binOp (.name "m") .bitXor (.name "n")) heap
      (number (m ^^^ n)) heap :=
    Evaluates.binary (Evaluates.lookup getM) (Evaluates.lookup getN)
      (by simp [Mutable.binary, scalarBitXor, number, integer])
  have parity : Evaluates program env
      (.binOp (.binOp (.name "m") .bitXor (.name "n")) .bitAnd (.constant (.int 1)))
      heap (number ((m ^^^ n) &&& 1)) heap :=
    Evaluates.binary xor Evaluates.constant
      (by simp [Mutable.binary, scalarBitAnd, number, integer])
  exact Evaluates.compare parity Evaluates.constant
    (by
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      norm_cast
      exact Nat.dvd_iff_mod_eq_zero)

structure Locals (env : HeapCalls.Env) (limit address m : Nat) : Prop where
  bound : env "limit" = some (number limit)
  array : env "counts" = some (.ref address)
  outer : env "m" = some (number m)
  maximum : env "m_max" = some (number (maxParameter limit))

theorem row_loop (limit m first count : Nat) (counts : Nat → Nat)
    (lower : 1 ≤ first) (endpoint : first+count=m)
    (base : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals env limit base.length m) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "n" first m 1 rowBody []]
      (arrayHeap base limit counts) finalEnv
      (arrayHeap base limit (row limit m first count counts)) ∧
      Locals finalEnv limit base.length m := by
  induction count generalizing first counts env with
  | zero =>
      refine ⟨env, ?_, locals⟩
      rw [row_zero]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "n" (number first)
      have boundLocals : Locals bound limit base.length m :=
        ⟨by simpa [bound, setLocal] using locals.bound,
          by simpa [bound, setLocal] using locals.array,
          by simpa [bound, setLocal] using locals.outer,
          by simpa [bound, setLocal] using locals.maximum⟩
      have parity := parity_test m first bound (arrayHeap base limit counts)
        boundLocals.outer (by simp [bound, setLocal])
      by_cases even : ((m ^^^ first) &&& 1) = 0
      · have rejected : permitted limit m first = false := by
          have reduced : (m-first)%2=0 := by
            simpa only [ProjectEulerStatements.PythagoreanPerimeterCounting.xor_parity m first
              (by omega)] using even
          simp [permitted, reduced]
        obtain ⟨finalEnv, rest, finalLocals⟩ := ih (first+1) counts (by omega) (by omega)
          bound boundLocals
        refine ⟨finalEnv, ?_, finalLocals⟩
        rw [row_step]
        simp only [rejected, Bool.false_eq_true, if_false]
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        apply BlockContinues.branch_next (flag := true)
          (by simpa only [even, decide_true, decide_false] using parity) rfl
        apply BlockContinues.continue_loop (by rfl)
        simpa only [Nat.cast_add, Nat.cast_one] using rest
      · have odd : (m-first)%2=1 := by
          have different := even
          rw [ProjectEulerStatements.PythagoreanPerimeterCounting.xor_parity m first
            (by omega)] at different
          omega
        have divisor := gcd_test m first bound (arrayHeap base limit counts)
          boundLocals.outer (by simp [bound, setLocal])
        by_cases bad : Nat.gcd m first ≠ 1
        · have rejected : permitted limit m first = false := by simp [permitted, bad]
          obtain ⟨finalEnv, rest, finalLocals⟩ := ih (first+1) counts (by omega) (by omega)
            bound boundLocals
          refine ⟨finalEnv, ?_, finalLocals⟩
          rw [row_step]
          simp only [rejected, Bool.false_eq_true, if_false]
          apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          apply BlockContinues.branch_next (flag := false)
            (by simpa only [even, decide_true, decide_false] using parity) rfl
          apply BlockContinues.branch_next (flag := true) (by simpa [bad] using divisor) rfl
          apply BlockContinues.continue_loop (by rfl)
          simpa only [Nat.cast_add, Nat.cast_one] using rest
        · let ready := setLocal bound "p0" (number (perimeter m first))
          have readyLocals : Locals ready limit base.length m :=
            ⟨by simpa [ready, setLocal] using boundLocals.bound,
              by simpa [ready, setLocal] using boundLocals.array,
              by simpa [ready, setLocal] using boundLocals.outer,
              by simpa [ready, setLocal] using boundLocals.maximum⟩
          have allocated : BlockContinues program bound [rowBody[2]'(by decide)]
              (arrayHeap base limit counts) ready (arrayHeap base limit counts) :=
            BlockContinues.assign (perimeter_value m first bound (arrayHeap base limit counts)
              boundLocals.outer (by simp [bound, setLocal]))
          have tested : Evaluates program ready
              (.compare (.name "p0") [(.gt, .name "limit")]) (arrayHeap base limit counts)
              (.scalar (.bool (decide (limit < perimeter m first))))
              (arrayHeap base limit counts) :=
            Evaluates.compare (Evaluates.lookup rfl) (Evaluates.lookup readyLocals.bound)
              (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
          by_cases large : limit < perimeter m first
          · refine ⟨ready, ?_, readyLocals⟩
            rw [row_inactive _ _ _ _ _ large]
            apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
            apply BlockContinues.branch_next (flag := false)
              (by simpa only [even, decide_true, decide_false] using parity) rfl
            apply BlockContinues.branch_next (flag := false) (by simpa [bad] using divisor) rfl
            apply BlockContinues.append allocated
            apply BlockContinues.branch_next (flag := true) (by simpa [large] using tested) rfl
            exact BlockContinues.break_loop (by rfl) .nil
          · have accepted : permitted limit m first = true := by
              simp [permitted, odd, Nat.coprime_iff_gcd_eq_one, not_ne_iff.mp bad,
                not_lt.mp large]
            have positive : 0 < perimeter m first := by
              exact Nat.mul_pos (Nat.mul_pos (by decide) (by omega)) (by omega)
            obtain ⟨middleEnv, executed, stable⟩ := multiple_for limit m first counts positive
              base ready readyLocals.array rfl readyLocals.bound
            have middleLocals : Locals middleEnv limit base.length m :=
              ⟨(stable "limit" (by decide)).trans readyLocals.bound,
                (stable "counts" (by decide)).trans readyLocals.array,
                (stable "m" (by decide)).trans readyLocals.outer,
                (stable "m_max" (by decide)).trans readyLocals.maximum⟩
            obtain ⟨finalEnv, rest, finalLocals⟩ := ih (first+1)
              (addMultiples limit m first 1 (limit/perimeter m first) counts)
              (by omega) (by omega) middleEnv middleLocals
            refine ⟨finalEnv, ?_, finalLocals⟩
            rw [row_step]
            simp only [accepted, if_true]
            apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
            apply BlockContinues.branch_next (flag := false)
              (by simpa only [even, decide_true, decide_false] using parity) rfl
            apply BlockContinues.branch_next (flag := false) (by simpa [bad] using divisor) rfl
            apply BlockContinues.append allocated
            apply BlockContinues.branch_next (flag := false) (by simpa [large] using tested) rfl
            apply BlockContinues.append executed
            apply BlockContinues.loop_resume
            simpa only [Nat.cast_add, Nat.cast_one] using rest

theorem outer_step_all (limit m count : Nat) (counts : Nat → Nat) :
    outer limit m (count+1) counts =
      outer limit (m+1) count (row limit m 1 (m-1) counts) := by
  rw [outer_step]
  by_cases active : 2*m*(m+1) ≤ limit
  · rw [if_pos active]
  · rw [if_neg active, row_inactive _ _ _ _ _ (by simpa [perimeter] using not_le.mp active)]

theorem outer_loop (limit first count : Nat) (counts : Nat → Nat)
    (lower : 2 ≤ first) (endpoint : first+count=maxParameter limit+1)
    (base : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit))
    (getArray : env "counts" = some (.ref base.length))
    (getMaximum : env "m_max" = some (number (maxParameter limit))) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "m" first (maxParameter limit+1) 1 outerBody []]
      (arrayHeap base limit counts) finalEnv
      (arrayHeap base limit (outer limit first count counts)) ∧
      finalEnv "limit" = some (number limit) ∧
      finalEnv "counts" = some (.ref base.length) ∧
      finalEnv "m_max" = some (number (maxParameter limit)) := by
  induction count generalizing first counts env with
  | zero =>
      refine ⟨env, ?_, getLimit, getArray, getMaximum⟩
      rw [outer_zero]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "m" (number first)
      have boundLocals : Locals bound limit base.length first :=
        ⟨by simpa [bound, setLocal] using getLimit,
          by simpa [bound, setLocal] using getArray,
          by simp [bound, setLocal], by simpa [bound, setLocal] using getMaximum⟩
      obtain ⟨middleEnv, rowRun, middleLocals⟩ := row_loop limit first 1 (first-1) counts
        (by omega) (by omega) base bound boundLocals
      have rowFor : BlockContinues program bound outerBody (arrayHeap base limit counts)
          middleEnv (arrayHeap base limit (row limit first 1 (first-1) counts)) := by
        apply BlockContinues.for_range (values := [number 1, number first])
          (.cons Evaluates.constant (.cons (Evaluates.lookup boundLocals.outer) .nil)) rfl
        exact rowRun
      obtain ⟨finalEnv, rest, finalLimit, finalArray, finalMaximum⟩ := ih (first+1)
        (row limit first 1 (first-1) counts) (by omega) (by omega) middleEnv
        middleLocals.bound middleLocals.array middleLocals.maximum
      refine ⟨finalEnv, ?_, finalLimit, finalArray, finalMaximum⟩
      rw [outer_step_all]
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Nat.cast_add, bound, number, integer] using rowFor.append rest

theorem maximum_value (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (number limit)) :
    Evaluates program env
      (.binOp (.call (.name "isqrt")
        [.binOp (.name "limit") .floorDiv (.constant (.int 2))]) .add (.constant (.int 1)))
      heap (number (maxParameter limit)) heap := by
  have divided : Evaluates program env
      (.binOp (.name "limit") .floorDiv (.constant (.int 2))) heap (number (limit/2)) heap :=
    Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number, integer])
  have rooted : Evaluates program env
      (.call (.name "isqrt") [.binOp (.name "limit") .floorDiv (.constant (.int 2))])
      heap (number (Nat.sqrt (limit/2))) heap := by
    apply Evaluates.library_call (.cons divided .nil) rfl
    have root_eq : Math.isqrt (limit/2) = Nat.sqrt (limit/2) :=
      Nat.eq_sqrt.mpr (Math.isqrt_spec (limit/2))
    change scalarResult (some (Math.isqrtCall [.int ((limit/2 : Nat) : Int)])) heap = _
    simp only [Math.isqrtCall, if_neg (show ¬ ((limit/2 : Nat) : Int) < 0 by omega),
      Int.toNat_natCast, root_eq, scalarResult]
    rfl
  exact Evaluates.binary rooted Evaluates.constant
    (by simp [Mutable.binary, scalarBinary, number, integer, maxParameter])

/-- Execute every sieve operation while keeping the caller's existing heap intact. -/
theorem sieve (limit : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) :
    ∃ finalEnv, BlockContinues program env (code.drop 1 |>.take 3) heap finalEnv
      (arrayHeap (heap ++ [.list [number 0]]) limit
        (ProjectEulerStatements.PythagoreanPerimeterCounting.table limit)) ∧
      finalEnv "counts" = some (.ref (heap.length+1)) := by
  let base := heap ++ [.list [number 0]]
  let allocated := setLocal env "counts" (.ref base.length)
  let ready := setLocal allocated "m_max" (number (maxParameter limit))
  have readyLimit : ready "limit" = some (number limit) := by
    simpa [ready, allocated, setLocal] using getLimit
  have readyArray : ready "counts" = some (.ref base.length) := by
    simp [ready, allocated, setLocal]
  obtain ⟨finalEnv, executed, finalLimit, finalArray, finalMaximum⟩ :=
    outer_loop limit 2 (Nat.sqrt (limit/2)) (fun _ => 0) (by decide)
      (by dsimp only [maxParameter]; omega) base ready readyLimit readyArray
      (by simp [ready, setLocal])
  rw [ProjectEulerStatements.PythagoreanPerimeterCounting.table_at_sqrt] at executed
  refine ⟨finalEnv, ?_, by simpa only [base, List.length_append, List.length_singleton]
    using finalArray⟩
  have initialized := initialize_result limit env heap getLimit
  have address : heap.length+1 = base.length := by simp [base]
  rw [address] at initialized
  apply BlockContinues.append (BlockContinues.annAssign initialized)
  apply BlockContinues.append (BlockContinues.assign
    (maximum_value limit allocated (arrayHeap base limit (fun _ => 0))
      (by simpa [allocated, setLocal] using getLimit)))
  apply BlockContinues.for_range
    (values := [number 2, number (maxParameter limit+1)])
    (.cons Evaluates.constant (.cons
      (Evaluates.binary (av := number (maxParameter limit)) (bv := number 1)
        (value := number (maxParameter limit+1))
        (middle := arrayHeap base limit (fun _ => 0))
        (afterRight := arrayHeap base limit (fun _ => 0))
        (finalHeap := arrayHeap base limit (fun _ => 0))
        (Evaluates.lookup (by simp [setLocal])) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number, integer])) .nil)) rfl
  exact executed

end solve
end solvers.«75»
