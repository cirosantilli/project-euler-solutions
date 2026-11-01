import solvers.«87.py.lean.d».Outer

namespace solvers.«87».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.PrimePowerTriples
set_option Elab.async false

def visited (limit : Nat) : List Nat :=
  prunedValues limit (powerLists limit).squares (powerLists limit).cubes (powerLists limit).fourths

theorem visited_complete (limit : Nat) : (visited limit).toFinset = exhaustive limit := by
  change (prunedValues limit (powerList limit 2 (primeCandidates limit))
    (powerList limit 3 (primeCandidates limit)) (powerList limit 4 (primeCandidates limit))).toFinset = _
  exact prunedValues_complete limit _ _ _
    (powerList_ordered limit 2 _ (primes_ordered _)) (powerList_ordered limit 3 _ (primes_ordered _))
    (powerList_complete limit 2) (powerList_complete limit 3) (powerList_complete limit 4)

theorem count_result (base : Mutable.Heap) (items : List Nat) (env : HeapCalls.Env)
    (getSet : env "representable" = some (.ref base.length)) :
    Evaluates program env (.builtinCall .len [.name "representable"])
      (search.working base items) (number items.toFinset.card) (search.working base items) := by
  apply Evaluates.builtin (values := [.ref base.length]) (.cons (Evaluates.lookup getSet) .nil)
  simp [primitive, Mutable.length, search.working, search.keys, number,
    ProjectEulerSolutions.ListEraseDups.length_eq_card]

theorem positive_result (limit : Nat) (positive : 0 < limit) (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "solve" [number limit] heap
      (number (exhaustive limit).card) (heap ++ extra) := by
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number limit] []
  obtain ⟨extra, primeAddress, preparedEnv, prepared, preparedReady, getSet⟩ := prepare limit positive heap initial rfl
  let base := heap ++ extra
  let tables := buildPowers.working base (powerLists limit)
  let addresses : search.Addresses := ⟨base.length, base.length + 1, base.length + 2⟩
  have ready : search.Locals preparedEnv limit addresses tables.length := by
    refine ⟨preparedReady.bound, preparedReady.squares, preparedReady.cubes, preparedReady.fourths, ?_⟩
    simpa [tables, base, buildPowers.working] using getSet
  obtain ⟨finalEnv, executed, finalReady⟩ := search.result limit
    (powerLists limit).squares (powerLists limit).cubes (powerLists limit).fourths tables addresses
    (by simp [tables, addresses, buildPowers.working])
    (by simp [tables, addresses, buildPowers.working])
    (by simp [tables, addresses, buildPowers.working]) [] preparedEnv ready
  have setup : BlockContinues program initial (code.take 10) heap preparedEnv (search.working tables []) := by
    simpa only [search.working, search.keys, List.eraseDups_nil, List.map_nil] using prepared
  have loop : BlockContinues program preparedEnv [code[10]'(by decide)] (search.working tables []) finalEnv
      (search.working tables (visited limit)) := by
    simpa only [List.nil_append, visited] using executed
  refine ⟨extra ++ [.list (encode (powerLists limit).squares), .list (encode (powerLists limit).cubes),
    .list (encode (powerLists limit).fourths), .set (search.keys (visited limit))], ?_⟩
  apply BlockReturns.function rfl rfl
  apply setup.returns
  apply loop.returns
  apply BlockReturns.return_value
  have returned := count_result tables (visited limit) finalEnv finalReady.set
  simpa [visited_complete, search.working, tables, base, buildPowers.working, List.append_assoc] using returned

/-- The entire optimized Python program returns the exact distinct count for every signed limit. -/
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "solve" [integer limit] heap
      (number (exhaustive limit.toNat).card) (heap ++ extra) := by
  by_cases nonpositive : limit ≤ 0
  · refine ⟨[], ?_⟩
    simpa [empty_nonpositive limit nonpositive] using nonpositive_result limit nonpositive heap
  · lift limit to Nat using (by omega : 0 ≤ limit)
    simpa only [Int.toNat_natCast] using positive_result limit (by omega) heap

end solvers.«87».solve
