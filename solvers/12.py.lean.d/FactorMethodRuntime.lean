import solvers.«12.py.lean.d».EnsureMethodRuntime
import solvers.«12.py.lean.d».FactorLoopRuntime

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

abbrev ensureBudget (bound : Nat) : Nat := (41*(bound+1)+150)*(bound+2)+140
abbrev factorBudget (bound m : Nat) : Nat :=
  ensureBudget bound + 80*(bound+1) + 40*m + PrimitiveWork.bitLength (m : Int) + 120

private theorem heap_set_same (heap : Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨valid, value⟩ := List.getElem?_eq_some_iff.mp found
  rw [← value, List.set_getElem_self valid]

/-- Count divisors using the actual mutable cache, retaining a single prime-size
bound for later calls in the search. -/
theorem factor_method_cost (m last bound : Nat) (positive : 0 < m)
    (primeBound : Nat.Prime bound) (enough : Nat.sqrt m+1 ≤ bound) (lastBound : last ≤ bound)
    (heap : Heap) (address storage : Nat) (ps : List Nat)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (lastEntry : ps.getLast? = some last) (ordered : ps.Pairwise (· < ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps (last+1)) :
    ∃ finalPrimes finalLast, finalPrimes.getLast? = some finalLast ∧ finalLast ≤ bound ∧
      finalPrimes.Pairwise (· < ·) ∧
      ProjectEulerStatements.PrimeTrialDivision.CompleteBelow finalPrimes (finalLast+1) ∧
      MethodReturnsIn .word program (.ref address) factorMethod.1 factorMethod.2 [.scalar (.int m)] heap
        (factorBudget bound m) (.scalar (.int m.divisors.card))
        (heap.set storage (.list (primeValues finalPrimes))) := by
  let env := functionEnv factorMethod.1 factorMethod.2 [.ref address, .scalar (.int m)] []
  have guard : EvaluatesIn .word program env (.compare (.name "m") [(.ltE, .constant (.int 1))]) heap 7
      (.scalar (.bool (decide (m ≤ 1)))) heap := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup rfl) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases small : m ≤ 1
  · have equal : m = 1 := by omega
    refine ⟨ps, last, lastEntry, lastBound, ordered, complete, ?_⟩
    rw [heap_set_same heap storage _ found]
    have body : BlockReturnsIn .word program env (executableBody factorMethod.2) heap 11
        (.scalar (.int m.divisors.card)) heap := by
      apply BlockReturnsIn.branch (cc := 7) (cb := 2) (flag := true) (by simpa [small] using guard) rfl
      simp only [equal, Nat.divisors_one, Finset.card_singleton]
      exact BlockReturnsIn.return_value EvaluatesIn.constant
    have run := MethodReturnsIn.of_body (receiver := .ref address) (values := [.scalar (.int m)]) rfl body
    intro anchors stack
    exact (run anchors stack).mono (by change 5+11 ≤ factorBudget bound m; simp only [factorBudget]; omega)
  · obtain ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalBound, finalOrder, finalComplete, ensured⟩ :=
      ensure_method_cost (Nat.sqrt m+1) last bound primeBound enough lastBound heap address storage ps
        object found lastEntry ordered complete
    let nextHeap := heap.set storage (.list (primeValues finalPrimes))
    have distinct : storage ≠ address := by rintro rfl; rw [object] at found; cases found
    have nextObject : nextHeap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]) := by
      simpa [nextHeap, distinct] using object
    have nextFound : nextHeap[storage]? = some (.list (primeValues finalPrimes)) := by
      simp [nextHeap, (List.getElem?_eq_some_iff.mp found).1]
    have primes (p : Nat) (member : p ∈ finalPrimes) : Nat.Prime p := ((finalComplete p).mp member).1
    have correct : ProjectEulerStatements.FactorDivisors.count m finalPrimes = m.divisors.card := by
      apply ProjectEulerStatements.FactorDivisors.count_correct m finalPrimes positive
        (finalOrder.imp (fun h => Nat.le_of_lt h)) primes
      intro p prime bounded _
      exact (finalComplete p).mpr ⟨prime, by omega⟩
    have lengthBound : finalPrimes.length ≤ bound+1 :=
      ProjectEulerStatements.PrimeCacheBounds.length_le_of_pairwise_lt_bounded finalPrimes bound finalOrder
        (fun p member => by have below := ((finalComplete p).mp member).2; omega)
    have root : Math.isqrt m = Nat.sqrt m := Nat.eq_sqrt.mpr (Math.isqrt_spec m)
    have library : ReturnsIn .word program "isqrt" [.scalar (.int m)] heap
        (PrimitiveWork.bitLength (m : Int)+1) (.scalar (.int (Nat.sqrt m))) heap := by
      apply ReturnsIn.library (values := [.int m]) rfl rfl
      · change scalarResult (some (Math.isqrtCall [.int m])) heap = _
        simp [scalarResult, Math.isqrtCall, root, show ¬ (m : Int) < 0 by omega]
      · rfl
    have calculated : EvaluatesIn .word program env (.call (.name "isqrt") [.name "m"]) heap
        (PrimitiveWork.bitLength (m : Int)+4) (.scalar (.int (Nat.sqrt m))) heap :=
      (EvaluatesIn.call (.cons (EvaluatesIn.lookup rfl) .nil) library).mono (by omega)
    have argument : EvaluatesIn .word program env
        (.binOp (.call (.name "isqrt") [.name "m"]) .add (.constant (.int 1))) heap
        (PrimitiveWork.bitLength (m : Int)+9) (.scalar (.int (Nat.sqrt m+1 : Nat))) heap := by
      exact (EvaluatesIn.binary (work := 1) calculated EvaluatesIn.constant
        (by simp [Mutable.binary, scalarBinary]) rfl).mono (by omega)
    have defined : program.findClass "PrimeTable" =
        some (origin, (program.findClassBody "PrimeTable").getD []) := rfl
    have method : findFunctionIn ((program.findClassBody "PrimeTable").getD []) "ensure_up_to" = some ensureMethod := rfl
    have boundMethod : bindMethod program (.ref address) "ensure_up_to" heap =
        some (.ok (.method (.ref address) ensureMethod.1 ensureMethod.2), heap) := by
      simp [bindMethod, object, instanceClass, defined, method]
      rfl
    have priced : RecordWork.method program heap (.ref address) "ensure_up_to" = some 20 := by
      simp [RecordWork.method, object, boundMethod, RecordWork.lookup]
      rfl
    have call := EvaluatesIn.method (name := "self") (env := env) rfl (by decide) boundMethod priced
      (ArgumentsIn.cons argument ArgumentsIn.nil) ensured
    let withX := setLocal env "x" (.scalar (.int m))
    let withTotal := setLocal withX "total" (.scalar (.int 1))
    have field := table_field nextHeap address storage withTotal nextObject rfl
    have loop := factor_loop m 1 positive nextHeap storage finalPrimes nextFound primes 0 withTotal rfl rfl
    have entered := BlockReturnsIn.for_list_field field nextFound loop
    have body : BlockReturnsIn .word program env (executableBody factorMethod.2) heap
        (ensureBudget bound+80*finalPrimes.length+40*m+PrimitiveWork.bitLength (m : Int)+109)
        (.scalar (.int m.divisors.card)) nextHeap := by
      apply BlockReturnsIn.mono
        (small := 7+((PrimitiveWork.bitLength (m : Int)+10+ensureBudget bound+20+3)+
          (1+(1+(17+(80*finalPrimes.length+40*m+40)+2)+2)+2)+2)+2)
        (larger := by omega)
      apply BlockReturnsIn.branch (cc := 7) (flag := false) (by simpa [small] using guard) rfl
      apply BlockReturnsIn.mono
        (small := (PrimitiveWork.bitLength (m : Int)+10+ensureBudget bound+20+3+2)+
          (1+(1+(17+(80*finalPrimes.length+40*m+40)+2)+2)+2)) (larger := by omega)
      apply (BlockContinuesIn.discard call).returns
      apply BlockReturnsIn.assign (EvaluatesIn.lookup rfl)
      apply BlockReturnsIn.assign EvaluatesIn.constant
      simpa only [List.drop_zero, Nat.one_mul, correct, Nat.sub_zero] using entered
    have run := MethodReturnsIn.of_body (receiver := .ref address) (values := [.scalar (.int m)]) rfl body
    refine ⟨finalPrimes, finalLast, finalEntry, finalBound, finalOrder, finalComplete, ?_⟩
    intro anchors stack
    apply (run anchors stack).mono
    change 5+(ensureBudget bound+80*finalPrimes.length+40*m+PrimitiveWork.bitLength (m : Int)+109) ≤ factorBudget bound m
    simp only [factorBudget]
    omega

end solvers.«12».runtimeCost
