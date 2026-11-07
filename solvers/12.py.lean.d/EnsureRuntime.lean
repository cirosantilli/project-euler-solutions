import solvers.«12.py.lean.d».PrimeRuntime
import lang.py.FormalSW.HeapRuntimeCostSetup

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

theorem table_last (heap : Mutable.Heap) (address : Nat) (ps : List Nat) (last : Nat)
    (table : TableView heap address ps) (lastEntry : ps.getLast? = some last)
    (env : HeapCalls.Env) (self : env "self" = some (.ref address)) :
    EvaluatesIn .word program env lastExpression heap 29 (.scalar (.int last)) heap := by
  obtain ⟨storage, object, found⟩ := table
  have nonempty : ps ≠ [] := by intro empty; simp [empty] at lastEntry
  have positive : 0 < ps.length := List.length_pos_iff.mpr nonempty
  have shifted : (-1 + (ps.length : Int)).toNat = ps.length - 1 := by omega
  have indexed : Mutable.index (primeValues ps) (.scalar (.int (-1))) heap =
      some (.ok (ps.length - 1), heap) := by
    simp [Mutable.index, primeValues, shifted, show ¬ (-1 + (ps.length : Int) < 0 ∨ (ps.length : Int) ≤ -1 + ps.length) by omega]
  have item : (primeValues ps)[ps.length - 1]? = some (.scalar (.int last)) := by
    rw [List.getLast?_eq_getElem?] at lastEntry
    simp only [primeValues, List.getElem?_map, lastEntry, Option.map_some]
  apply EvaluatesIn.list_subscript (by intros; simp)
    (table_field heap address storage env object self)
    (EvaluatesIn.negate_int EvaluatesIn.constant) found
  simp [Mutable.getItem, found, indexed, Except.map, item]

theorem prime_call (n : Nat) (heap : Heap) (address storage : Nat) (ps : List Nat) (env : HeapCalls.Env)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (self : env "self" = some (.ref address))
    (getN : env "candidate" = some (.scalar (.int n)))
    (large : 2 ≤ n) (ordered : ps.Pairwise (· ≤ ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps n) :
    EvaluatesIn .word program env
      (.call (.attribute (.name "self") "_is_prime") [.name "candidate"]) heap
      (40*ps.length+PrimitiveWork.bitLength (n : Int)+59)
      (.scalar (.bool (decide (Nat.Prime n)))) heap := by
  have defined : program.findClass "PrimeTable" =
      some (origin, (program.findClassBody "PrimeTable").getD []) := rfl
  have method : findFunctionIn ((program.findClassBody "PrimeTable").getD []) "_is_prime" = some primeMethod := rfl
  have bound : bindMethod program (.ref address) "_is_prime" heap =
      some (.ok (.method (.ref address) primeMethod.1 primeMethod.2), heap) := by
    simp [bindMethod, object, instanceClass, defined, method]
    rfl
  have priced : RecordWork.method program heap (.ref address) "_is_prime" = some 17 := by
    simp [RecordWork.method, object, bound, RecordWork.lookup]
    rfl
  have positive (p : Nat) (member : p ∈ ps) : 0 < p := ((complete p).mp member).1.pos
  have run := prime_method_cost n heap address storage ps object found positive
  rw [ProjectEulerStatements.PrimeTrialDivision.trial_complete n ps large ordered complete] at run
  exact (EvaluatesIn.method self (by decide) bound priced
    (ArgumentsIn.cons (EvaluatesIn.lookup getN) ArgumentsIn.nil) run).mono (by omega)

private theorem heap_set_same (heap : Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨valid, value⟩ := List.getElem?_eq_some_iff.mp found
  rw [← value, List.set_getElem_self valid]

theorem ensure_step (candidate : Nat) (heap : Mutable.Heap) (address storage : Nat)
    (ps : List Nat) (env : HeapCalls.Env)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (self : env "self" = some (.ref address))
    (getCandidate : env "candidate" = some (.scalar (.int candidate)))
    (large : 2 ≤ candidate) (ordered : ps.Pairwise (· ≤ ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps candidate) :
    BlockContinuesIn .word program env ensureIteration heap
      (40*ps.length+PrimitiveWork.bitLength (candidate : Int)+100)
      (HeapCalls.setLocal env "candidate" (.scalar (.int (candidate + 2 : Nat))))
      (heap.set storage (.list (primeValues (if Nat.Prime candidate then ps ++ [candidate] else ps)))) := by
  have tested := prime_call candidate heap address storage ps env object found self getCandidate large ordered complete
  have first : BlockContinuesIn .word program env [ensureIteration[0]'(by decide)] heap (40*ps.length+PrimitiveWork.bitLength (candidate : Int)+93) env
      (heap.set storage (.list (primeValues (if Nat.Prime candidate then ps ++ [candidate] else ps)))) := by
    by_cases prime : Nat.Prime candidate
    · rw [if_pos prime]
      apply BlockContinuesIn.mono (small := (40*ps.length+PrimitiveWork.bitLength (candidate : Int)+59)+27+2) (larger := by omega)
      apply BlockContinuesIn.branch (cc := 40*ps.length+PrimitiveWork.bitLength (candidate : Int)+59) (cb := 27) (flag := true) (by simpa [prime] using tested) rfl
      apply BlockContinuesIn.discard
      simpa only [primeValues, List.map_append, List.map_cons, List.map_nil] using
        EvaluatesIn.append (table_field heap address storage env object self)
          found (EvaluatesIn.lookup getCandidate) found
    · rw [if_neg prime, heap_set_same heap storage _ found]
      apply BlockContinuesIn.mono (small := (40*ps.length+PrimitiveWork.bitLength (candidate : Int)+59)+0+2) (larger := by omega)
      exact BlockContinuesIn.branch (cc := 40*ps.length+PrimitiveWork.bitLength (candidate : Int)+59) (cb := 0) (flag := false) (by simpa [prime] using tested) rfl
        BlockContinuesIn.nil
  have advanced : BlockContinuesIn .word program env [.augAssign (.name "candidate") .add (.constant (.int 2))]
      (heap.set storage (.list (primeValues (if Nat.Prime candidate then ps ++ [candidate] else ps)))) 7
      (setLocal env "candidate" (.scalar (.int (candidate+2 : Nat))))
      (heap.set storage (.list (primeValues (if Nat.Prime candidate then ps ++ [candidate] else ps)))) := by
    apply BlockContinuesIn.augAssign (work := 1) getCandidate EvaluatesIn.constant
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
    · rfl
  exact (first.append advanced).mono (by omega)


end solvers.«12».runtimeCost
