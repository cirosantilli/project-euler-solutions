import «.formalsw».solvers.«47.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p47.py»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.ConsecutiveFactors

set_option Elab.async false

namespace solvers.«47»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P47Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

namespace Sieve
open ProjectEulerSolutions.PrimeFactorSieve

/-- A source list whose entries are the current mathematical sieve values. -/
def table (size : Nat) (value : Nat → Nat) : List Mutable.Value :=
  (List.range size).map (fun n => number (value n))

@[simp] theorem table_length (size : Nat) (value : Nat → Nat) : (table size value).length = size := by
  simp [table]

theorem table_lookup (size : Nat) (value : Nat → Nat) (position : Nat) (inside : position < size) :
    (table size value)[position]? = some (number (value position)) := by
  simp [table, inside]

theorem table_set (size : Nat) (value : Nat → Nat) (position replacement : Nat) (inside : position < size) :
    (table size value).set position (number replacement) =
      table size (fun n => if n = position then replacement else value n) := by
  apply List.ext_getElem?
  intro n
  by_cases bounded : n < size
  · by_cases same : position = n
    · subst n
      rw [List.getElem?_set_self (by simpa using inside), table_lookup size _ position inside]
      simp
    · rw [List.getElem?_set_ne same, table_lookup size value n bounded,
        table_lookup size _ n bounded]
      simp [Ne.symm same]
  · have outside : size ≤ n := by omega
    simp only [List.getElem?_eq_none (by simpa using outside : ((table size value).set position (number replacement)).length ≤ n),
      List.getElem?_eq_none (by simpa using outside : (table size (fun k => if k = position then replacement else value k)).length ≤ n)]

private def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "distinct_prime_factor_counts").getD ([], [])

private def outerBody : List Stmt :=
  match ((executableBody definition.2)[1]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private def innerBody : List Stmt :=
  match (outerBody[0]? : Option Stmt) with
  | some (.ifThen _ [.forIn _ _ body _] _) => body
  | _ => []

private structure Locals (env : Mutable.Env) (maximum address : Nat) : Prop where
  bound : env "limit" = some (number maximum)
  array : env "cnt" = some (.ref address)

/-- The source augmented subscription performs one exact in-place table update. -/
private theorem increment (size j address : Nat) (inside : j < size) (value : Nat → Nat)
    (env : Mutable.Env) (heap : Mutable.Heap)
    (array : env "cnt" = some (.ref address)) (index : env "m" = some (number j))
    (found : heap[address]? = some (.list (table size value))) :
    BlockContinues program env innerBody heap env
      (heap.set address (.list (table size (fun n => if n = j then value j + 1 else value n)))) := by
  apply BlockContinues.augItem (by intro a b c; simp) (Evaluates.lookup array) (Evaluates.lookup index)
    (previous := number (value j)) (updated := number (value j + 1))
    (afterRead := heap) (afterOperation := heap) (rhs := Evaluates.constant)
  · have item := table_lookup size value j inside
    have bound : j < (table size value).length := by simpa using inside
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using Mutable.getItem_list heap address j _ found bound
  · simp [inplace, Mutable.binary, scalarBinary, number]
  · have nonnegative : ¬ (j : Int) < 0 := by omega
    have bounded : ¬ size ≤ j := by omega
    rw [← table_set size value j (value j + 1) inside]
    simp [Mutable.setItem, Mutable.index, found, nonnegative, bounded, number, Except.map]

theorem table_congr (size : Nat) (f g : Nat → Nat) (equal : ∀ n, n < size → f n = g n) :
    table size f = table size g := by
  apply List.map_congr_left
  intro n member
  rw [equal n (List.mem_range.mp member)]

private def arrayHeap (base : Mutable.Heap) (size : Nat) (value : Nat → Nat) : Mutable.Heap :=
  base ++ [.list (table size value)]

private theorem arrayHeap_store (base : Mutable.Heap) (size : Nat) (f g : Nat → Nat) :
    (arrayHeap base size f).set base.length (.list (table size g)) = arrayHeap base size g := by
  simp [arrayHeap, List.set_append_right]


/-- Each prime pass visits every multiple once and performs an in-place increment. -/
private theorem inner_loop (maximum p m : Nat) (prime : Nat.Prime p)
    (lower : p≤m) (multiple : p∣m) (env : Mutable.Env) (base : Mutable.Heap)
    (locals : Locals env maximum base.length) (getP : env "p"=some (number p)) :
    ∃ finalEnv, Locals finalEnv maximum base.length ∧ finalEnv "p"=some (number p) ∧
      BlockContinues program env [.rangeStepLoop "m" m (maximum+1) p innerBody []]
        (arrayHeap base (maximum+1) (row (countsBefore p) p m)) finalEnv
        (arrayHeap base (maximum+1) (countsBefore (p+1))) := by
  have positive := prime.pos
  by_cases active : m<maximum+1
  · let bound := setLocal env "m" (number m)
    have boundLocals : Locals bound maximum base.length :=
      ⟨by simpa [bound, setLocal] using locals.bound, by simpa [bound, setLocal] using locals.array⟩
    have boundP : bound "p"=some (number p) := by simpa [bound, setLocal] using getP
    obtain ⟨finalEnv, finalLocals, finalP, rest⟩ := inner_loop maximum p (m+p) prime (by omega)
      (Nat.dvd_add multiple (dvd_refl p)) bound base boundLocals boundP
    refine ⟨finalEnv, finalLocals, finalP, ?_⟩
    apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    apply BlockContinues.append (body := innerBody) (middleEnv := bound)
      (middleHeap := arrayHeap base (maximum+1) (row (countsBefore p) p (m+p)))
    · have updated := increment (maximum+1) m base.length active (row (countsBefore p) p m)
        bound (arrayHeap base (maximum+1) (row (countsBefore p) p m)) boundLocals.array
        (by simp [bound, setLocal]) (by simp [arrayHeap])
      have values : (fun n => if n=m then row (countsBefore p) p m m+1 else row (countsBefore p) p m n)=
          row (countsBefore p) p (m+p) := by
        funext n
        by_cases same : n=m
        · subst n
          simpa using (row_step (countsBefore p) p m m positive lower multiple).symm
        · simpa only [if_neg same] using (row_step (countsBefore p) p m n positive lower multiple).symm
      simpa only [arrayHeap_store, values] using updated
    · simpa [Calls.loopContinuation, innerBody] using rest
  · have complete : arrayHeap base (maximum+1) (row (countsBefore p) p m)=
        arrayHeap base (maximum+1) (countsBefore (p+1)) := by
      exact congrArg (fun items => base++[Mutable.Object.list items])
        (table_congr (maximum+1) _ _ (fun n bound => row_complete p m n prime (by omega)))
    refine ⟨env, locals, getP, ?_⟩
    rw [← complete]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues, positive]; omega) .nil
termination_by maximum+1-m
decreasing_by omega

/-- Zero-count primality detection is proved from the partially completed sieve. -/
private theorem prime_pass (maximum p : Nat) (lower : 2≤p) (inside : p≤maximum)
    (env : Mutable.Env) (base : Mutable.Heap) (locals : Locals env maximum base.length)
    (getP : env "p"=some (number p)) :
    ∃ finalEnv, Locals finalEnv maximum base.length ∧
      BlockContinues program env outerBody (arrayHeap base (maximum+1) (countsBefore p)) finalEnv
        (arrayHeap base (maximum+1) (countsBefore (p+1))) := by
  let heap := arrayHeap base (maximum+1) (countsBefore p)
  have read : Evaluates program env (.subscript (.name "cnt") (.name "p"))
      heap (number (countsBefore p p)) heap := by
    apply Evaluates.subscript (by intro a b c h; cases h) (Evaluates.lookup locals.array) (Evaluates.lookup getP)
    have item := table_lookup (maximum+1) (countsBefore p) p (by omega)
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using
      Mutable.getItem_list heap base.length p (table (maximum+1) (countsBefore p)) (by simp [heap, arrayHeap]) (by simp; omega)
  have tested : Evaluates program env (.compare (.subscript (.name "cnt") (.name "p")) [(.eq, .constant (.int 0))])
      heap (flag (decide (Nat.Prime p))) heap := by
    apply Evaluates.compare (av := number (countsBefore p p)) (bv := number 0) (middle := heap) (afterRight := heap)
      read Evaluates.constant
    change some (Except.ok (ε := ExceptionObject) (flag (decide (((countsBefore p p : Nat) : Int)=0))), heap)=_
    simp only [Int.natCast_eq_zero, current_zero p lower]
  by_cases prime : Nat.Prime p
  · obtain ⟨finalEnv, finalLocals, _, scanned⟩ := inner_loop maximum p p prime (by omega) (dvd_refl p) env base locals getP
    have initial : row (countsBefore p) p p=countsBefore p := funext (row_start _ p)
    rw [initial] at scanned
    refine ⟨finalEnv, finalLocals, ?_⟩
    apply BlockContinues.branch (flag := true) (by simpa only [prime, decide_true] using tested) rfl
    apply BlockContinues.for_range (values := [number p, number (maximum+1), number p])
      (start := p) (stop := maximum+1) (stride := p) (middle := heap)
      (.cons (Evaluates.lookup getP) (.cons
        (Evaluates.binary (av := number maximum) (bv := number 1) (middle := heap) (afterRight := heap) (finalHeap := heap)
          (Evaluates.lookup locals.bound) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number]))
        (.cons (Evaluates.lookup getP) .nil)))
    · change (if (p : Int)=0 then .error { typeName := "ValueError", args := [] }
          else .ok (↑p, ↑(maximum+1), ↑p) : Except ExceptionObject (Int×Int×Int))=_
      simp [prime.ne_zero]
    · exact scanned
  · have same : countsBefore p=countsBefore (p+1) := funext (skip_composite p · prime)
    refine ⟨env, locals, ?_⟩
    rw [← same]
    apply BlockContinues.branch (flag := false) (by simpa only [prime, decide_false] using tested) rfl
    exact .nil

private theorem outer_loop (count first maximum : Nat) (lower : 2≤first)
    (stop : first+count=maximum+1) (env : Mutable.Env) (base : Mutable.Heap)
    (locals : Locals env maximum base.length) :
    ∃ finalEnv, Locals finalEnv maximum base.length ∧
      BlockContinues program env [.rangeStepLoop "p" first (maximum+1) 1 outerBody []]
        (arrayHeap base (maximum+1) (countsBefore first)) finalEnv
        (arrayHeap base (maximum+1) (countsBefore (maximum+1))) := by
  induction count generalizing first env with
  | zero =>
      have same : first=maximum+1 := by omega
      refine ⟨env, locals, ?_⟩
      rw [same]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil
  | succ count ih =>
      let bound := setLocal env "p" (number first)
      have boundLocals : Locals bound maximum base.length :=
        ⟨by simpa [bound, setLocal] using locals.bound, by simpa [bound, setLocal] using locals.array⟩
      obtain ⟨middleEnv, middleLocals, passed⟩ := prime_pass maximum first lower (by omega) bound base boundLocals rfl
      obtain ⟨finalEnv, finalLocals, rest⟩ := ih (first+1) (by omega) (by omega) middleEnv middleLocals
      refine ⟨finalEnv, finalLocals, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.append passed
      simpa [Calls.loopContinuation, outerBody] using rest

private theorem initialize_list (maximum : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (bound : env "limit"=some (number maximum)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) heap (.ref (heap.length+1))
      (arrayHeap (heap++[.list [number 0]]) (maximum+1) (countsBefore 2)) := by
  let base := heap++[Mutable.Object.list [number 0]]
  apply Evaluates.binary (av := .ref heap.length) (bv := number (maximum+1)) (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.binary (Evaluates.lookup bound) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
  · simp [Mutable.binary, show (BinOp.mult==BinOp.mult)=true from rfl,
      Mutable.array, Mutable.allocate, number, base, arrayHeap, table, Except.map]

/-- The actual source sieve computes every distinct-prime-factor count in its table. -/
theorem result_positive (maximum : Nat) (positive : 0<maximum) (heap : Mutable.Heap) :
    ReturnsValue program "distinct_prime_factor_counts" [number maximum] heap (.ref (heap.length+1))
      (heap++[.list [number 0], .list (table (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors)]) := by
  let base := heap++[Mutable.Object.list [number 0]]
  let entry := functionEnv definition.1 definition.2 [number maximum] []
  let initial := setLocal entry "cnt" (.ref base.length)
  have initialLocals : Locals initial maximum base.length := ⟨rfl, by simp [initial, setLocal]⟩
  obtain ⟨finalEnv, finalLocals, scanned⟩ := outer_loop (maximum-1) 2 maximum (by decide) (by omega) initial base initialLocals
  have completed : table (maximum+1) (countsBefore (maximum+1))=
      table (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors :=
    table_congr (maximum+1) _ _ (fun n bound => countsBefore_complete _ n bound)
  have executed : ReturnsValue program "distinct_prime_factor_counts" [number maximum] heap (.ref base.length)
      (arrayHeap base (maximum+1) (countsBefore (maximum+1))) := by
    apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    apply BlockReturns.assign (value := .ref base.length)
    · simpa only [base, List.length_append, List.length_cons, List.length_nil, Nat.add_zero] using
        initialize_list maximum entry heap rfl
    · apply BlockContinues.returns (body := [.forIn (.name "p") (.call (.name "range")
        [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))]) outerBody []])
        (middleEnv := finalEnv) (middleHeap := arrayHeap base (maximum+1) (countsBefore (maximum+1)))
      · apply BlockContinues.for_range (values := [number 2, number (maximum+1)])
          (start := 2) (stop := maximum+1) (stride := 1) (middle := arrayHeap base (maximum+1) (countsBefore 2))
          (.cons Evaluates.constant (.cons
            (Evaluates.binary (av := number maximum) (bv := number 1)
              (Evaluates.lookup initialLocals.bound) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number, base])) .nil)) rfl
        exact scanned
      · exact BlockReturns.return_value (Evaluates.lookup finalLocals.array)
  simpa only [arrayHeap, completed, base, List.append_assoc, List.cons_append, List.nil_append,
    List.length_append, List.length_cons, List.length_nil, Nat.add_zero] using executed

end Sieve
namespace Scan
open ProjectEulerSolutions.ConsecutiveFactors
private abbrev code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private abbrev outerBody : List Stmt :=
  match (code[1]? : Option Stmt) with | some (.whileDo _ body _) => body | _ => []
private abbrev body : List Stmt :=
  match (outerBody[2]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private def condition : Expr := .compare (.subscript (.name "cnt") (.name "n")) [(.eq, .name "k")]
private def finished : Expr := .compare (.name "streak") [(.eq, .name "run_len")]

private structure Locals (env : HeapCalls.Env) (k maximum address : Nat) : Prop where
  factors : env "k"=some (number k)
  length : env "run_len"=some (number k)
  array : env "cnt"=some (.ref address)
  bound : env "limit"=some (number maximum)

private theorem Locals.bind (locals : Locals env k maximum address) (name : String) (value : Mutable.Value)
    (notK : name≠"k") (notLength : name≠"run_len") (notArray : name≠"cnt") (notLimit : name≠"limit") :
    Locals (setLocal env name value) k maximum address := by
  exact ⟨by simpa [setLocal, Ne.symm notK] using locals.factors,
    by simpa [setLocal, Ne.symm notLength] using locals.length,
    by simpa [setLocal, Ne.symm notArray] using locals.array,
    by simpa [setLocal, Ne.symm notLimit] using locals.bound⟩

private theorem tested (k maximum n address : Nat) (within : n≤maximum)
    (heap : Mutable.Heap) (found : heap[address]?=some (.list (Sieve.table (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors)))
    (env : HeapCalls.Env) (locals : Locals env k maximum address) (getN : env "n"=some (number n)) :
    Evaluates program env condition heap (flag (good k n)) heap := by
  apply Evaluates.compare (av := number (ProjectEulerStatements.P47.numDistinctPrimeFactors n))
    (bv := number k) (middle := heap) (afterRight := heap)
  · apply Evaluates.subscript (by intro a b c h; cases h) (Evaluates.lookup locals.array) (Evaluates.lookup getN)
    have item := Sieve.table_lookup (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors n (by omega)
    simpa only [(List.getElem?_eq_some_iff.mp item).2] using
      Mutable.getItem_list heap address n (Sieve.table (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors)
        found (by simp; omega)
  · exact Evaluates.lookup locals.factors
  · change some (Except.ok (ε := ExceptionObject)
      (flag (decide (((ProjectEulerStatements.P47.numDistinctPrimeFactors n : Nat) : Int)=(k : Int)))), heap)=_
    simp only [Int.natCast_inj, good]

private theorem finished_test (k length : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLength : env "run_len"=some (number k)) (getStreak : env "streak"=some (number length)) :
    Evaluates program env finished heap (flag (decide (length=k))) heap := by
  apply Evaluates.compare (av := number length) (bv := number k) (middle := heap) (afterRight := heap)
    (Evaluates.lookup getStreak) (Evaluates.lookup getLength)
  change some (Except.ok (ε := ExceptionObject) (flag (decide ((length : Int)=(k : Int)))), heap)=_
  simp only [Int.natCast_inj]

/-- Before the first full run, one source iteration updates only its streak local. -/
private theorem step_before (k maximum n address : Nat) (earlier : n<lastIndex k) (within : n≤maximum)
    (heap : Mutable.Heap) (found : heap[address]?=some (.list (Sieve.table (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors)))
    (env : HeapCalls.Env) (locals : Locals env k maximum address) (getN : env "n"=some (number n))
    (getStreak : env "streak"=some (number (streak (good k) n))) :
    ∃ finalEnv, Locals finalEnv k maximum address ∧ finalEnv "streak"=some (number (streak (good k) (n+1))) ∧
      BlockContinues program env body heap finalEnv heap := by
  have compared := tested k maximum n address within heap found env locals getN
  by_cases accepted : good k n=true
  · let length := streak (good k) n+1
    let updated := setLocal env "streak" (number length)
    have nextStreak : streak (good k) (n+1)=length := by simp only [streak, accepted, if_true, length]
    have short : length<k := by
      rw [← nextStreak]
      apply streak_before k (n+1)
      dsimp [lastIndex] at earlier
      omega
    have updatedLocals : Locals updated k maximum address := locals.bind "streak" (number length) (by decide) (by decide) (by decide) (by decide)
    refine ⟨updated, updatedLocals, by simp [updated, setLocal, nextStreak], ?_⟩
    apply BlockContinues.branch (flag := true) (by simpa only [accepted] using compared) rfl
    apply BlockContinues.append (body := [.augAssign (.name "streak") .add (.constant (.int 1))])
      (middleEnv := updated) (middleHeap := heap)
    · apply BlockContinues.augAssign (value := number length) (middle := heap) (finalHeap := heap)
        getStreak Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, number, length]
    · apply BlockContinues.branch (flag := false)
        (by simpa only [show length≠k by omega, decide_false] using finished_test k length heap updated updatedLocals.length rfl) rfl
      exact .nil
  · have rejected : good k n=false := by cases value : good k n <;> simp_all
    let updated := setLocal env "streak" (number 0)
    refine ⟨updated, locals.bind "streak" (number 0) (by decide) (by decide) (by decide) (by decide), ?_, ?_⟩
    · simp [updated, setLocal, streak, rejected]
    · apply BlockContinues.branch (flag := false) (by simpa only [rejected] using compared) rfl
      exact BlockContinues.assign Evaluates.constant

/-- A bounded scan entirely before the winning right boundary completes normally. -/
private theorem before_loop (k maximum first count stop address : Nat) (length : first+count=stop)
    (earlier : stop≤lastIndex k) (bounded : stop≤maximum+1)
    (heap : Mutable.Heap) (found : heap[address]?=some (.list (Sieve.table (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors)))
    (env : HeapCalls.Env) (locals : Locals env k maximum address)
    (getStreak : env "streak"=some (number (streak (good k) first))) :
    ∃ finalEnv, Locals finalEnv k maximum address ∧ finalEnv "streak"=some (number (streak (good k) stop)) ∧
      BlockContinues program env [.rangeStepLoop "n" first stop 1 body []] heap finalEnv heap := by
  induction count generalizing first env with
  | zero =>
      have same : first=stop := by omega
      refine ⟨env, locals, by simpa only [same] using getStreak, ?_⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "n" (number first)
      have boundLocals : Locals bound k maximum address := locals.bind "n" (number first) (by decide) (by decide) (by decide) (by decide)
      obtain ⟨updated, updatedLocals, updatedStreak, stepped⟩ := step_before k maximum first address (by omega) (by omega)
        heap found bound boundLocals rfl (by simpa [bound, setLocal] using getStreak)
      obtain ⟨finalEnv, finalLocals, finalStreak, rest⟩ := ih (first+1) (by omega) updated updatedLocals updatedStreak
      refine ⟨finalEnv, finalLocals, finalStreak, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockContinues.append stepped
      simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

/-- A scan reaching the winning boundary returns the least run's start immediately. -/
private theorem hit_loop (k maximum first address : Nat) (positive : 0<k)
    (before : first≤lastIndex k) (covered : lastIndex k≤maximum)
    (heap : Mutable.Heap) (found : heap[address]?=some (.list (Sieve.table (maximum+1) ProjectEulerStatements.P47.numDistinctPrimeFactors)))
    (following : List Stmt) (env : HeapCalls.Env) (locals : Locals env k maximum address)
    (getStreak : env "streak"=some (number (streak (good k) first))) :
    BlockReturns program env (.rangeStepLoop "n" first (maximum+1) 1 body [] :: following)
      heap (number (least k)) heap := by
  let bound := setLocal env "n" (number first)
  have boundLocals : Locals bound k maximum address := locals.bind "n" (number first) (by decide) (by decide) (by decide) (by decide)
  have boundStreak : bound "streak"=some (number (streak (good k) first)) := by simpa [bound, setLocal] using getStreak
  by_cases earlier : first<lastIndex k
  · obtain ⟨updated, updatedLocals, updatedStreak, stepped⟩ := step_before k maximum first address earlier (by omega)
      heap found bound boundLocals rfl boundStreak
    have rest := hit_loop k maximum (first+1) address positive (by omega) covered heap found following updated updatedLocals updatedStreak
    apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
    apply stepped.returns
    simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest
  · have same : first=lastIndex k := by omega
    have accepted : good k first=true := same ▸ winner_good k positive
    have full : streak (good k) first+1=k := by
      have reached := streak_winner k positive
      simpa only [← same, streak, accepted, if_true] using reached
    let updated := setLocal bound "streak" (number k)
    have updatedLocals : Locals updated k maximum address := boundLocals.bind "streak" (number k) (by decide) (by decide) (by decide) (by decide)
    have start := least_lower k positive
    have indexLarge : k≤first := by dsimp [lastIndex] at same; omega
    have answer : first-k+1=least k := by dsimp [lastIndex] at same; omega
    apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
    apply BlockReturns.branch (flag := true)
      (by simpa only [accepted] using tested k maximum first address (by omega) heap found bound boundLocals rfl) rfl
    apply BlockReturns.augAssign (value := number k) (middle := heap) (afterOperation := heap)
      boundStreak Evaluates.constant
    · have numeric : ((streak (good k) first : Nat) : Int)+1=(k : Int) := by exact_mod_cast full
      simp [inplace, Mutable.binary, scalarBinary, number, numeric]
    · apply BlockReturns.branch (flag := true)
        (by simpa only [decide_true] using finished_test k k heap updated updatedLocals.length rfl) rfl
      apply BlockReturns.return_value
      apply Evaluates.binary (av := number (first-k)) (bv := number 1) (middle := heap) (afterRight := heap)
      · exact Evaluates.binary (av := number first) (bv := number k)
          (Evaluates.lookup (by simp [bound, setLocal])) (Evaluates.lookup updatedLocals.length)
          (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub indexLarge])
      · exact Evaluates.constant
      · simp [Mutable.binary, scalarBinary, number, ← answer]
termination_by lastIndex k-first

decreasing_by omega
end Scan

namespace solve
open ProjectEulerSolutions.ConsecutiveFactors
private abbrev body : List Stmt := Scan.outerBody
private def rangeExpression : Expr := .call (.name "range")
  [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))]

private theorem loop (k limit : Nat) (positive : 0<k) (positiveLimit : 0<limit)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getK : env "k"=some (number k))
    (getLength : env "run_len"=some (number k)) (getLimit : env "limit"=some (number limit)) :
    ∃ finalHeap, BlockReturns program env [.whileDo (.constant (.bool true)) body []]
      heap (number (least k)) finalHeap := by
  let allocated := heap++[Mutable.Object.list [number 0],
    .list (Sieve.table (limit+1) ProjectEulerStatements.P47.numDistinctPrimeFactors)]
  let address := heap.length+1
  let listed := setLocal env "cnt" (.ref address)
  let ready := setLocal listed "streak" (number 0)
  have locals : Scan.Locals ready k limit address :=
    ⟨by simpa [ready, listed, setLocal] using getK,
      by simpa [ready, listed, setLocal] using getLength,
      by simp [ready, listed, setLocal],
      by simpa [ready, listed, setLocal] using getLimit⟩
  have found : allocated[address]?=some (.list (Sieve.table (limit+1) ProjectEulerStatements.P47.numDistinctPrimeFactors)) := by
    simp [allocated, address]
  have zero : ready "streak"=some (number (streak (good k) 2)) := by
    simp [ready, setLocal, streak_two k positive]
  have bounds : Evaluates.Arguments program ready [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))]
      allocated [number 2, number (limit+1)] allocated :=
    .cons Evaluates.constant (.cons
      (Evaluates.binary (av := number limit) (bv := number 1) (middle := allocated) (afterRight := allocated) (finalHeap := allocated)
        (Evaluates.lookup locals.bound) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil)
  have sieve : Evaluates program env (.call (.name "distinct_prime_factor_counts") [.name "limit"])
      heap (.ref address) allocated :=
    Evaluates.call (.cons (Evaluates.lookup getLimit) .nil) (Sieve.result_positive limit positiveLimit heap)
  by_cases covered : lastIndex k≤limit
  · refine ⟨allocated, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply BlockReturns.assign sieve
    apply BlockReturns.assign Evaluates.constant
    apply BlockReturns.for_range (values := [number 2, number (limit+1)]) bounds rfl
    have start : 2≤lastIndex k := by
      have := least_lower k positive
      dsimp [lastIndex]
      omega
    exact Scan.hit_loop k limit 2 address positive start covered allocated found _ ready locals zero
  · obtain ⟨finalEnv, finalLocals, _, scanned⟩ := Scan.before_loop k limit 2 (limit-1) (limit+1) address
      (by omega) (by omega) (by omega) allocated found ready locals zero
    have rangeScan : BlockContinues program ready [.forIn (.name "n") rangeExpression Scan.body []]
        allocated finalEnv allocated := by
      apply BlockContinues.for_range (values := [number 2, number (limit+1)]) bounds rfl
      exact scanned
    let next := setLocal finalEnv "limit" (number (limit*2))
    obtain ⟨finalHeap, rest⟩ := loop k (limit*2) positive (by omega) allocated next
      (by simpa [next, setLocal] using finalLocals.factors)
      (by simpa [next, setLocal] using finalLocals.length) rfl
    refine ⟨finalHeap, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply BlockReturns.assign sieve
    apply BlockReturns.assign Evaluates.constant
    apply rangeScan.returns
    apply BlockReturns.augAssign (value := number (limit*2)) (middle := allocated) (afterOperation := allocated)
      finalLocals.bound Evaluates.constant
    · simp [inplace, Mutable.binary, scalarBinary, number]
    · simpa [body, Scan.outerBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest
termination_by lastIndex k-limit

decreasing_by omega

/-- Every positive initial bound eventually reaches the first diagonal run. -/
theorem result (k initialLimit : Nat) (positive : 0<k) (positiveLimit : 0<initialLimit) :
    ∃ finalHeap, ReturnsValue program "solve" [number k, number k, number initialLimit] []
      (number (least k)) finalHeap := by
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number k, number k, number initialLimit] []
  let ready := setLocal initial "limit" (number initialLimit)
  obtain ⟨finalHeap, searched⟩ := loop k initialLimit positive positiveLimit [] ready
    (by simp [ready, setLocal, initial]; rfl) (by simp [ready, setLocal, initial]; rfl) rfl
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (Evaluates.lookup rfl)
  exact searched
end solve

namespace Domain
private abbrev inputEnv (k runLength initialLimit : Int) : ScalarEnv :=
  fun name => if name="k" then some (.int k) else if name="run_len" then some (.int runLength)
    else if name="start_limit" then some (.int initialLimit) else none

private theorem evaluated (k runLength initialLimit : Int) :
    Calls.Evaluates program (inputEnv k runLength initialLimit) Solvers.P47Ast.requires_solve_0
      (.bool (decide (1≤k ∧ runLength=k ∧ 1≤initialLimit))) := by
  have factors : Calls.Evaluates program (inputEnv k runLength initialLimit)
      (.compare (.name "k") [(.gtE, .constant (.int 1))]) (.bool (decide (1≤k))) :=
    Calls.Evaluates.compare (Calls.Evaluates.lookup rfl) Calls.Evaluates.constant rfl
  have length : Calls.Evaluates program (inputEnv k runLength initialLimit)
      (.compare (.name "run_len") [(.eq, .name "k")]) (.bool (decide (runLength=k))) :=
    Calls.Evaluates.compare (Calls.Evaluates.lookup rfl) (Calls.Evaluates.lookup rfl) rfl
  have initial : Calls.Evaluates program (inputEnv k runLength initialLimit)
      (.compare (.name "start_limit") [(.gtE, .constant (.int 1))]) (.bool (decide (1≤initialLimit))) :=
    Calls.Evaluates.compare (Calls.Evaluates.lookup rfl) Calls.Evaluates.constant rfl
  by_cases positive : 1≤k
  · apply Calls.Evaluates.boolean_continue (by simpa only [positive, decide_true] using factors) rfl
    by_cases same : runLength=k
    · apply Calls.Evaluates.boolean_continue (by simpa only [same, decide_true] using length) rfl
      apply Calls.Evaluates.boolean_last
      simpa only [positive, same, true_and] using initial
    · simp only [same, and_false, false_and, decide_false]
      exact Calls.Evaluates.boolean_stop (by simpa only [same, decide_false] using length) rfl
  · simp only [positive, false_and, decide_false]
    exact Calls.Evaluates.boolean_stop (by simpa only [positive, decide_false] using factors) rfl

theorem bounds (k runLength initialLimit : Int)
    (condition : Calls.Condition program (inputEnv k runLength initialLimit) Solvers.P47Ast.requires_solve_0) :
    1≤k ∧ runLength=k ∧ 1≤initialLimit := by
  simpa using (evaluated k runLength initialLimit).condition_iff.mp condition
end Domain

theorem solve.«Equiv(naive_solver)» (k runLength initialLimit : Int)
    (condition : Calls.Condition program (Domain.inputEnv k runLength initialLimit) Solvers.P47Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int k, .int runLength, .int initialLimit]).Equiv
      (HeapCalls.scalarFunction P47Ast.module "naive_solver" [.int k, .int runLength, .int initialLimit]) := by
  obtain ⟨positive, same, positiveLimit⟩ := Domain.bounds k runLength initialLimit condition
  subst runLength
  lift k to Nat using (by omega : 0≤k)
  lift initialLimit to Nat using (by omega : 0≤initialLimit)
  obtain ⟨_, returned⟩ := solve.result k initialLimit (by omega) (by omega)
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p47.naive_solver.result k k initialLimit []
  have optimized := returned.scalar_finishes (values := [.int k, .int k, .int initialLimit])
  have original := reference.scalar_finishes (values := [.int k, .int k, .int initialLimit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem solve.«Supported» :
    (∀ k runLength initialLimit : Int,
      Calls.Condition program (Domain.inputEnv k runLength initialLimit) Solvers.P47Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int k, .int runLength, .int initialLimit]).Supported) ∧
    (∀ k runLength initialLimit : Int,
      Calls.Condition program (Domain.inputEnv k runLength initialLimit) Solvers.P47Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P47Ast.module "naive_solver" [.int k, .int runLength, .int initialLimit]).Supported) := by
  constructor
  · intro k runLength initialLimit condition
    obtain ⟨positive, same, positiveLimit⟩ := Domain.bounds k runLength initialLimit condition
    subst runLength
    lift k to Nat using (by omega : 0≤k)
    lift initialLimit to Nat using (by omega : 0≤initialLimit)
    obtain ⟨_, returned⟩ := solve.result k initialLimit (by omega) (by omega)
    exact (returned.scalar_returns (values := [.int k, .int k, .int initialLimit])).supported
  · intro k runLength initialLimit _
    exact data.«project-euler-statements».data.py.project_euler_statements.p47.naive_solver.«Supported» k runLength initialLimit

end solvers.«47»
