import ProjectEulerSolutions.Lib.PrimeSieve
import Mathlib.Data.List.Range

/-! Boolean-list sieves share the prime-divisor invariant of the bytearray sieve. -/
namespace ProjectEulerSolutions.BooleanPrimeSieve
open PrimeSieve

set_option Elab.async false

def clearRange (flags : List Bool) (start stop step : Nat) : List Bool :=
  (List.range' start ((stop-start+step-1)/step) step).foldl (fun result i => result.set i false) flags

theorem clearRange_length (flags : List Bool) (start stop step : Nat) :
    (clearRange flags start stop step).length=flags.length := by
  unfold clearRange
  generalize List.range' start ((stop-start+step-1)/step) step=indices
  induction indices generalizing flags with
  | nil => rfl
  | cons i rest ih => simpa only [List.foldl_cons, List.length_set] using ih (flags.set i false)

theorem count_next (start stop step : Nat) (positive : 0<step) (active : start<stop) :
    (stop-start+step-1)/step=(stop-(start+step)+step-1)/step+1 := by
  rw [show stop-start+step-1=(stop-start-1)+step by omega, Nat.add_div_right _ positive]
  by_cases near : stop≤start+step
  · simp [show stop-(start+step)=0 by omega,
      Nat.div_eq_of_lt (show stop-start-1<step by omega), Nat.div_eq_of_lt (show step-1<step by omega)]
  · congr 1
    congr 1
    omega

theorem clearRange_step (flags : List Bool) (start stop step : Nat) (positive : 0<step) (active : start<stop) :
    clearRange flags start stop step=clearRange (flags.set start false) (start+step) stop step := by
  simp only [clearRange, count_next start stop step positive active, List.range'_succ, List.foldl_cons]

theorem clearRange_done (flags : List Bool) (start stop step : Nat) (positive : 0<step) (finished : stop≤start) :
    clearRange flags start stop step=flags := by
  simp [clearRange, Nat.sub_eq_zero_of_le finished, Nat.div_eq_of_lt (show step-1<step by omega)]

def mark (flags : List Bool) (p : Nat) : List Bool := clearRange flags (p*p) flags.length p

@[simp] theorem mark_length (flags : List Bool) (p : Nat) : (mark flags p).length=flags.length := clearRange_length ..

theorem mark_get (flags : List Bool) (p i : Nat) (positive : 0<p) (valid : i<flags.length) :
    (mark flags p)[i]?=if p^2≤i ∧ p∣i then some false else flags[i]? := by
  rw [mark, clearRange, FormalSW.Py.Builtin.List.getElem?_setUniform _ _ _ _ valid]
  change (if i ∈ markedIndices flags.length p then some false else flags[i]?)=_
  simp only [mem_markedIndices _ _ _ positive, valid, true_and]

def Sieved (limit processed : Nat) (flags : List Bool) : Prop :=
  flags.length=limit+1 ∧ flags[0]?=some false ∧ flags[1]?=some false ∧
    ∀ i flag, 2≤i → flags[i]?=some flag → (flag=true ↔ Survives processed i)

def initial (limit : Nat) : List Bool := [false,false]++List.replicate (limit-1) true

theorem initial_stores (limit : Nat) (large : 2≤limit) :
    ((List.replicate (limit+1) true).set 0 false).set 1 false=initial limit := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le large
  rw [show 2+n+1=(n+1)+1+1 by omega]
  change false :: false :: List.replicate (n+1) true=initial (2+n)
  simp [initial, show 2+n-1=n+1 by omega]

theorem sieved_initial (limit : Nat) (large : 2≤limit) : Sieved limit 2 (initial limit) := by
  refine ⟨by simp [initial]; omega, by simp [initial], by simp [initial], ?_⟩
  intro i flag lower found
  have equal : flag=true := by
    have tail : (List.replicate (limit-1) true)[i-2]?=some flag := by
      rw [show i=(i-2)+2 by omega] at found
      simpa [initial] using found
    exact List.eq_of_mem_replicate (List.mem_of_getElem? tail)
  simp only [equal, true_iff]
  intro p before prime
  have := prime.two_le
  omega

theorem sieved_skip (limit p : Nat) (flags : List Bool) (sieved : Sieved limit p flags)
    (composite : ¬ Nat.Prime p) : Sieved limit (p+1) flags := by
  refine ⟨sieved.1, sieved.2.1, sieved.2.2.1, ?_⟩
  intro i flag lower found
  rw [sieved.2.2.2 i flag lower found, survives_next]
  simp [composite]

theorem sieved_mark (limit p : Nat) (flags : List Bool) (sieved : Sieved limit p flags)
    (prime : Nat.Prime p) : Sieved limit (p+1) (mark flags p) := by
  have size := sieved.1
  have pLower := prime.two_le
  refine ⟨by simpa using size, ?_, ?_, ?_⟩
  · rw [mark_get flags p 0 prime.pos (by rw [size]; omega)]
    simp [sieved.2.1, show ¬ p^2≤0 by nlinarith]
  · rw [mark_get flags p 1 prime.pos (by have := (List.getElem?_eq_some_iff.mp sieved.2.2.1).1; omega)]
    simp [sieved.2.2.1, show ¬ p^2≤1 by nlinarith]
  · intro i flag lower found
    have valid : i<flags.length := by simpa using (List.getElem?_eq_some_iff.mp found).1
    rw [mark_get flags p i prime.pos valid] at found
    rw [survives_next]
    by_cases removed : p^2≤i ∧ p∣i
    · simp only [if_pos removed, Option.some.injEq] at found
      subst flag
      simp [prime, removed.1, removed.2]
    · simp only [if_neg removed] at found
      rw [sieved.2.2.2 i flag lower found]
      simp only [prime, true_implies]
      tauto

/-- Crossing the square-root boundary makes every cell its exact primality Boolean. -/
theorem finished (limit p : Nat) (flags : List Bool) (sieved : Sieved limit p flags) (stopped : limit<p^2) :
    flags=(List.range (limit+1)).map (fun n => decide (Nat.Prime n)) := by
  have size := sieved.1
  apply List.ext_getElem?
  intro i
  by_cases within : i<flags.length
  · let flag := flags[i]
    have found : flags[i]?=some flag := List.getElem?_eq_getElem within
    have inside : i<limit+1 := by omega
    rw [found]
    simp only [List.getElem?_map, List.getElem?_range inside, Option.map_some, Option.some.injEq]
    by_cases lower : 2≤i
    · exact Bool.eq_iff_iff.mpr (by
        simp only [decide_eq_true_eq]
        exact (sieved.2.2.2 i flag lower found).trans (survives_finished limit p i lower (by omega) stopped))
    · have small : i=0 ∨ i=1 := by omega
      rcases small with rfl | rfl
      · have equal : flag=false := Option.some.inj (found.symm.trans sieved.2.1)
        simpa only [equal] using (by decide : false=decide (Nat.Prime 0))
      · have equal : flag=false := Option.some.inj (found.symm.trans sieved.2.2.1)
        simpa only [equal] using (by decide : false=decide (Nat.Prime 1))
  · have outside : limit+1≤i := by omega
    rw [List.getElem?_eq_none (by omega : flags.length≤i)]
    simp [List.getElem?_eq_none (l := List.range (limit+1)) (by simpa using outside)]

end ProjectEulerSolutions.BooleanPrimeSieve
