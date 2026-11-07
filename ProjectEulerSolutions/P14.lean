import ProjectEulerStatements.Lib.Collatz
import ProjectEulerStatements.Lib.ArgMaximum
namespace ProjectEulerSolutions.P14
open ProjectEulerStatements.P14 ProjectEulerStatements.Collatz

/-- A cache covers both small and large visited values. Zero denotes a miss. -/
structure Cache where
  values : Nat → Nat
  valid : ValidCache values

def initialCache : Cache := ⟨fun n => if n = 1 then 1 else 0, initialCache_valid⟩

def store (cache : Cache) (n length : Nat) (correct : length = collatzLen n) : Cache :=
  ⟨Function.update cache.values n length, by rw [correct]; exact cache.valid.update n⟩

/-- Follow uncached links and cache their lengths while unwinding the path. -/
def memoLength (n : Nat) (cache : Cache) : { result : Nat × Cache // result.1 = collatzLen n } :=
  if zero : n = 0 then ⟨(0, cache), by simp [zero]⟩
  else if one : n = 1 then ⟨(1, cache), by simp [one]⟩
  else if hit : cache.values n ≠ 0 then
    ⟨(cache.values n, cache), (cache.valid n).hit hit⟩
  else
    let suffix := memoLength (collatzStep n) cache
    let length := suffix.val.1 + 1
    have correct : length = collatzLen n := unwind_length n suffix.val.1 (by omega) one suffix.property
    ⟨(length, store suffix.val.2 n length correct), correct⟩
termination_by collatzMeasure n
decreasing_by exact collatzMeasure_decreases (by omega) (by assumption)

def scan : List Nat → Nat → Cache → Nat
  | [], best, _ => best
  | n :: rest, best, cache =>
      let current := memoLength n cache
      let previous := memoLength best current.val.2
      scan rest (if previous.val.1 < current.val.1 then n else best) previous.val.2

/-- Include the supplied limit, matching the Python call to longest_collatz_under(limit+1). -/
def solve (limit : Nat) : Nat := scan (List.range' 1 limit) 0 initialCache
end ProjectEulerSolutions.P14
