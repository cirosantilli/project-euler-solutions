import solvers.«87.py.lean.d».Powers

namespace solvers.«87»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

namespace solve
abbrev code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
def primeCandidates (limit : Nat) := ProjectEulerStatements.PrimePartitions.primesUpTo (Nat.sqrt limit + 1)
def powerLists (limit : Nat) : buildPowers.Lists := buildPowers.extend limit (primeCandidates limit) ⟨[], [], []⟩
def initialArrays (env : HeapCalls.Env) (first : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "squares" (.ref first)) "cubes" (.ref (first + 1))) "fourths" (.ref (first + 2))

theorem nonpositive_result (limit : Int) (nonpositive : limit ≤ 0) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [integer limit] heap (number 0) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := true) (value := .scalar (.bool true)) (middle := heap)
  · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonpositive]
  · rfl
  · exact BlockReturns.return_value Evaluates.constant

theorem setup_lists (limit : Nat) (base : Mutable.Heap) (primeAddress : Nat)
    (found : base[primeAddress]? = some (.list (encode (primeCandidates limit))))
    (env : HeapCalls.Env) (getLimit : env "limit" = some (number limit))
    (getPrimes : env "primes" = some (.ref primeAddress)) :
    ∃ finalEnv, BlockContinues program env ((code.drop 2).take 4) base finalEnv
      (buildPowers.working base (powerLists limit)) ∧ buildPowers.Locals finalEnv limit primeAddress base.length := by
  let e1 := setLocal env "squares" (.ref base.length)
  let e2 := setLocal e1 "cubes" (.ref (base.length + 1))
  let ready := initialArrays env base.length
  let h1 := base ++ [.list []]
  let h2 := base ++ [.list [], .list []]
  have first : BlockContinues program env [code[2]'(by decide)] base e1 h1 :=
    BlockContinues.annAssign (Evaluates.list .nil)
  have second : BlockContinues program e1 [code[3]'(by decide)] h1 e2 h2 := by
    apply BlockContinues.annAssign
    simpa [h1, h2, List.append_assoc] using (Evaluates.list (module := program) (env := e1) (heap := h1) .nil)
  have third : BlockContinues program e2 [code[4]'(by decide)] h2 ready (buildPowers.working base ⟨[], [], []⟩) := by
    apply BlockContinues.annAssign
    simpa [h2, buildPowers.working, encode, List.append_assoc] using
      (Evaluates.list (module := program) (env := e2) (heap := h2) .nil)
  have locals : buildPowers.Locals ready limit primeAddress base.length := by
    constructor
    · simpa [ready, initialArrays, setLocal] using getLimit
    · simpa [ready, initialArrays, setLocal] using getPrimes
    · simp [ready, initialArrays, setLocal]
    · simp [ready, initialArrays, setLocal]
    · simp [ready, initialArrays, setLocal]
  obtain ⟨finalEnv, executed, finalReady⟩ := buildPowers.result limit (primeCandidates limit) base primeAddress found
    ⟨[], [], []⟩ ready locals
  exact ⟨finalEnv, first.append (second.append (third.append executed)), finalReady⟩

private theorem sort_ordered (heap : Mutable.Heap) (address : Nat) (items : List Nat)
    (found : heap[address]? = some (.list (encode items))) (ordered : items.Pairwise (· ≤ ·)) :
    primitive (.sort address) [] heap = some (.ok (.scalar .none), heap) := by
  let integers := items.map (fun (n : Nat) => (n : Int))
  let keys := items.map (fun (n : Nat) => Constant.int n)
  have orderedInts : integers.Pairwise (· ≤ ·) :=
    ordered.map (fun (n : Nat) => (n : Int)) (fun _ _ lower => Int.ofNat_le.mpr lower)
  have sorted : ScalarSort.ints integers = integers :=
    List.Perm.eq_of_pairwise (fun _ _ _ _ lower upper => le_antisymm lower upper)
      (ScalarSort.ints_ordered _) orderedInts (ScalarSort.ints_perm _)
  have keysSorted : ScalarSort.sort keys = .ok keys := by
    have result := ScalarSort.sort_ints integers
    rw [sorted] at result
    simpa only [integers, keys, List.map_map, Function.comp_def] using result
  have same : heap.set address (.list (encode items)) = heap := by
    have bound := (List.getElem?_eq_some_iff.mp found).1
    have equal := (List.getElem?_eq_some_iff.mp found).2
    rw [← equal, List.set_getElem_self bound]
  have operation := primitive_sort_list heap address keys keys (by simpa [keys, encode, number, List.map_map] using found) keysSorted
  have encoded : keys.map Mutable.Value.scalar = encode items := by simp [keys, encode, number, List.map_map]
  simpa only [encoded, same] using operation

private theorem sort_name (heap : Mutable.Heap) (address : Nat) (items : List Nat) (env : HeapCalls.Env)
    (name : String) (getName : env name = some (.ref address))
    (found : heap[address]? = some (.list (encode items))) (ordered : items.Pairwise (· ≤ ·)) :
    BlockContinues program env [.expr (.call (.attribute (.name name) "sort") [])] heap env heap := by
  apply BlockContinues.discard
  apply Evaluates.sort (Evaluates.lookup getName) found (sort_ordered heap address items found ordered)
  intro localName same
  cases same
  simp [getName]

theorem sort_lists (limit : Nat) (base : Mutable.Heap) (primeAddress : Nat) (env : HeapCalls.Env)
    (ready : buildPowers.Locals env limit primeAddress base.length) :
    BlockContinues program env ((code.drop 6).take 3) (buildPowers.working base (powerLists limit)) env
      (buildPowers.working base (powerLists limit)) := by
  let heap := buildPowers.working base (powerLists limit)
  have ordered (exponent : Nat) : (powerList limit exponent (primeCandidates limit)).Pairwise (· ≤ ·) :=
    powerList_ordered limit exponent _ (primes_ordered _)
  have squares := sort_name heap base.length (powerList limit 2 (primeCandidates limit)) env "squares" ready.squares
    (by simp [heap, buildPowers.working, powerLists, buildPowers.extend]) (ordered 2)
  have cubes := sort_name heap (base.length + 1) (powerList limit 3 (primeCandidates limit)) env "cubes" ready.cubes
    (by simp [heap, buildPowers.working, powerLists, buildPowers.extend]) (ordered 3)
  have fourths := sort_name heap (base.length + 2) (powerList limit 4 (primeCandidates limit)) env "fourths" ready.fourths
    (by simp [heap, buildPowers.working, powerLists, buildPowers.extend]) (ordered 4)
  exact squares.append (cubes.append fourths)

/-- The source prepares complete sorted power lists and a fresh empty set before searching. -/
theorem prepare (limit : Nat) (positive : 0 < limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) :
    ∃ extra primeAddress finalEnv,
      BlockContinues program env (code.take 10) heap finalEnv
        (buildPowers.working (heap ++ extra) (powerLists limit) ++ [.set []]) ∧
      buildPowers.Locals finalEnv limit primeAddress (heap ++ extra).length ∧
      finalEnv "representable" = some (.ref ((heap ++ extra).length + 3)) := by
  have guard : BlockContinues program env [code[0]'(by decide)] heap env heap := by
    apply BlockContinues.branch (flag := false) (value := .scalar (.bool false)) (middle := heap)
    · apply Evaluates.compare (Evaluates.lookup getLimit) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, positive.ne']
    · rfl
    · exact .nil
  have root : Evaluates program env (.call (.name "isqrt") [.name "limit"]) heap (number (Nat.sqrt limit)) heap := by
    apply Evaluates.library_call (.cons (Evaluates.lookup getLimit) .nil) rfl
    change scalarResult (some (Math.isqrtCall [.int limit])) heap = _
    have same : Math.isqrt limit = Nat.sqrt limit := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
    simp [scalarResult, Math.isqrtCall, same, number, show ¬ (limit : Int) < 0 by omega]
  obtain ⟨primeAddress, extra, primes, found⟩ := sieve_primes.result (Nat.sqrt limit + 1 : Nat) heap
  let base := heap ++ extra
  let primeEnv := setLocal env "primes" (.ref primeAddress)
  have assigned : BlockContinues program env [code[1]'(by decide)] heap primeEnv base := by
    apply BlockContinues.assign
    apply Evaluates.call (middle := heap) (values := [number (Nat.sqrt limit + 1)])
    · exact .cons (Evaluates.binary (middle := heap) (afterRight := heap) (av := number (Nat.sqrt limit)) (bv := number 1)
        root Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil
    · exact primes
  obtain ⟨listEnv, built, ready⟩ := setup_lists limit base primeAddress (by simpa [primeCandidates] using found)
    primeEnv (by simpa [primeEnv, setLocal] using getLimit) (by simp [primeEnv, setLocal])
  let tables := buildPowers.working base (powerLists limit)
  let finalEnv := setLocal listEnv "representable" (.ref (base.length + 3))
  have allocated : BlockContinues program listEnv [code[9]'(by decide)] tables finalEnv (tables ++ [.set []]) := by
    apply BlockContinues.assign
    apply Evaluates.builtin (values := []) Evaluates.Arguments.nil
    have emptySet : primitive (.builtin .set) [] tables = some (.ok (.ref tables.length), tables ++ [.set []]) := rfl
    simpa [tables, buildPowers.working] using emptySet
  refine ⟨extra, primeAddress, finalEnv, ?_, ?_, by simp [finalEnv, setLocal, base]⟩
  · exact guard.append (assigned.append (built.append ((sort_lists limit base primeAddress listEnv ready).append allocated)))
  · exact ready.set_other "representable" (.ref (base.length + 3)) (by decide) (by decide) (by decide) (by decide) (by decide)

end solve
end solvers.«87»
