import lang.py.FormalSW.KeyDictionaryCalls
import ProjectEulerSolutions.Lib.HeapCollection
import Mathlib.Data.List.Nodup

/-! Tuple pattern keys retain their representatives and shared lists of square strings. -/
namespace ProjectEulerSolutions.PatternBuckets
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

abbrev Key := List Nat
def number (n : Nat) : Mutable.Value := .scalar (.int n)
def text (value : String) : Mutable.Value := .scalar (.string value)
def texts (items : List String) : List Mutable.Value := items.map text
def hash (key : Key) : HashKey := .tuple (key.map fun (n : Nat) => .scalar (.int n))

structure Bucket where
  key : Key
  representative : Nat
  address : Nat
  values : List String
deriving Repr, DecidableEq

def entry (bucket : Bucket) : HashDict.Entry Mutable.Value :=
  (hash bucket.key, .ref bucket.representative, .ref bucket.address)
def entries (buckets : List Bucket) := buckets.map entry
def object : List Bucket → Mutable.Object
  | [] => .dict []
  | first :: rest => .keyDict (entries (first :: rest))
def find (buckets : List Bucket) (key : Key) : Option Bucket :=
  buckets.find? (fun bucket => decide (bucket.key = key))
def chosen (buckets : List Bucket) (key : Key) (fresh : Nat) : Nat :=
  ((find buckets key).map Bucket.address).getD fresh
def contents (buckets : List Bucket) (key : Key) : List String :=
  ((find buckets key).map Bucket.values).getD []

/-- Existing keys keep both object identities; a missing key receives the fresh default list. -/
def push : List Bucket → Key → Nat → Nat → String → List Bucket
  | [], key, representative, address, value => [⟨key, representative, address, [value]⟩]
  | bucket :: rest, key, representative, address, value =>
      if bucket.key = key then { bucket with values := bucket.values ++ [value] } :: rest
      else bucket :: push rest key representative address value

private theorem scalar_keys_equal (fuel : Nat) (positive : 0 < fuel) (left right : Key) :
    (left.map fun (n : Nat) => HashKey.scalar (.int n)).isEqv (right.map fun (n : Nat) => HashKey.scalar (.int n))
      (HashKey.equalFuel fuel) = decide (left = right) := by
  cases fuel with
  | zero => omega
  | succ fuel =>
      induction left generalizing right with
      | nil => cases right <;> rfl
      | cons a rest ih =>
          cases right with
          | nil => rfl
          | cons b tail => simp [List.isEqv, HashKey.equalFuel, scalarCompare, evalCmpOp, ih, Int.natCast_inj]

theorem hash_equal (left right : Key) : HashKey.equal (hash left) (hash right) = decide (left = right) := by
  unfold HashKey.equal hash
  simp only [HashKey.depth, HashKey.equalFuel]
  apply scalar_keys_equal
  omega

theorem lookup_entries (buckets : List Bucket) (key : Key) :
    HashDict.lookup (entries buckets) (hash key) = (find buckets key).map (fun bucket => .ref bucket.address) := by
  induction buckets with
  | nil => rfl
  | cons bucket rest ih =>
      simp only [entries, List.map_cons, HashDict.lookup_cons, entry, hash_equal]
      by_cases same : bucket.key = key
      · simp [find, same]
      · simpa only [find, List.find?_cons, show decide (bucket.key = key) = false by simp [same],
          if_false] using ih

theorem entries_push (buckets : List Bucket) (key : Key) (representative address : Nat) (value : String) :
    entries (push buckets key representative address value) =
      HashDict.insert (entries buckets) (hash key) (.ref representative) (.ref (chosen buckets key address)) := by
  induction buckets with
  | nil => rfl
  | cons bucket rest ih =>
      by_cases same : bucket.key = key
      · simp [push, same, entries, entry, HashDict.insert, hash_equal, chosen, find]
      · simpa [push, same, entries, entry, HashDict.insert, hash_equal, chosen, find] using ih

theorem push_nonempty (buckets : List Bucket) (key : Key) (representative address : Nat) (value : String) :
    push buckets key representative address value ≠ [] := by
  cases buckets with
  | nil => simp [push]
  | cons bucket rest => by_cases same : bucket.key = key <;> simp [push, same]

theorem object_push (buckets : List Bucket) (key : Key) (representative address : Nat) (value : String) :
    object (push buckets key representative address value) =
      .keyDict (entries (push buckets key representative address value)) := by
  cases changed : push buckets key representative address value with
  | nil => exact (push_nonempty buckets key representative address value changed).elim
  | cons first rest => rfl

theorem entries_push_hit (buckets : List Bucket) (key : Key) (representative address : Nat) (value : String)
    (bucket : Bucket) (found : find buckets key = some bucket) :
    entries (push buckets key representative address value) = entries buckets := by
  induction buckets with
  | nil => simp [find] at found
  | cons first rest ih =>
      by_cases same : first.key = key
      · simp [push, same, entries, entry]
      · have tail : find rest key = some bucket := by simpa [find, same] using found
        simpa only [push, if_neg same, entries, List.map_cons] using congrArg (List.cons (entry first)) (ih tail)

theorem contents_push (buckets : List Bucket) (key query : Key) (representative address : Nat) (value : String) :
    contents (push buckets key representative address value) query =
      if query = key then contents buckets key ++ [value] else contents buckets query := by
  induction buckets with
  | nil => by_cases same : key = query <;> simp [push, contents, find, same, Ne.symm]
  | cons first rest ih =>
      by_cases updated : first.key = key <;> by_cases selected : first.key = query <;>
        by_cases same : query = key <;> simp_all [push, contents, find]

theorem push_miss (buckets : List Bucket) (key : Key) (representative address : Nat) (value : String)
    (missing : find buckets key = none) :
    push buckets key representative address value = buckets ++ [⟨key, representative, address, [value]⟩] := by
  induction buckets with
  | nil => rfl
  | cons first rest ih =>
      have different : first.key ≠ key := by
        intro same
        simp [find, same] at missing
      have tail : find rest key = none := by simpa [find, different] using missing
      simp [push, different, ih tail]

theorem push_split (before after : List Bucket) (selected : Bucket) (key : Key)
    (matched : selected.key = key) (skipped : ∀ bucket ∈ before, bucket.key ≠ key)
    (representative address : Nat) (value : String) :
    push (before ++ selected :: after) key representative address value =
      before ++ { selected with values := selected.values ++ [value] } :: after := by
  induction before with
  | nil => simp [push, matched]
  | cons first rest ih =>
      have different := skipped first List.mem_cons_self
      simp only [List.cons_append, push, if_neg different]
      rw [ih (fun bucket member => skipped bucket (List.mem_cons_of_mem _ member))]

theorem addresses_push (buckets : List Bucket) (key : Key) (representative address : Nat) (value : String) :
    (push buckets key representative address value).map Bucket.address =
      if (find buckets key).isSome then buckets.map Bucket.address else buckets.map Bucket.address ++ [address] := by
  induction buckets with
  | nil => rfl
  | cons first rest ih =>
      by_cases same : first.key = key
      · simp [push, same, find]
      · have skipped : find (first :: rest) key = find rest key := by simp [find, same]
        simp only [push, if_neg same, List.map_cons, ih, skipped]
        split <;> simp_all

theorem keys_push (buckets : List Bucket) (key : Key) (representative address : Nat) (value : String) :
    (push buckets key representative address value).map Bucket.key =
      if (find buckets key).isSome then buckets.map Bucket.key else buckets.map Bucket.key ++ [key] := by
  induction buckets with
  | nil => rfl
  | cons first rest ih =>
      by_cases same : first.key = key
      · simp [push, same, find]
      · have skipped : find (first :: rest) key = find rest key := by simp [find, same]
        simp only [push, if_neg same, List.map_cons, ih, skipped]
        split <;> simp_all

theorem find_of_mem (buckets : List Bucket) (bucket : Bucket) (member : bucket ∈ buckets)
    (unique : (buckets.map Bucket.key).Nodup) : find buckets bucket.key = some bucket := by
  induction buckets with
  | nil => simp at member
  | cons first rest ih =>
      have distinct := List.nodup_cons.mp unique
      rcases List.mem_cons.mp member with rfl | member
      · simp [find]
      · have different : first.key ≠ bucket.key := by
          intro same
          apply distinct.1
          rw [same]
          exact List.mem_map_of_mem member
        simpa [find, different] using ih member distinct.2

/-- The heap addresses are independent of the exact mathematical partition of the input. -/
structure Partition (keyOf : String → Key) (items : List String) (buckets : List Bucket) : Prop where
  unique : (buckets.map Bucket.key).Nodup
  contents : ∀ key, contents buckets key = items.filter (fun value => decide (keyOf value = key))

theorem Partition.empty (keyOf : String → Key) : Partition keyOf [] [] := by
  constructor
  · simp
  · intro key
    rfl

theorem Partition.push {keyOf : String → Key} {items : List String} {buckets : List Bucket}
    (partition : Partition keyOf items buckets) (representative address : Nat) (value : String) :
    Partition keyOf (items ++ [value]) (push buckets (keyOf value) representative address value) := by
  constructor
  · rw [keys_push]
    cases selected : find buckets (keyOf value) with
    | some bucket => simpa only [selected, Option.isSome_some, if_true] using partition.unique
    | none =>
        have missing : keyOf value ∉ buckets.map Bucket.key := by
          intro member
          obtain ⟨bucket, inside, same⟩ := List.mem_map.mp member
          have found := find_of_mem buckets bucket inside partition.unique
          rw [same, selected] at found
          cases found
        simpa only [selected, Option.isSome_none, Bool.false_eq_true, if_false, List.concat_eq_append] using
          List.Nodup.concat missing partition.unique
  · intro key
    rw [contents_push, partition.contents, partition.contents, List.filter_append]
    by_cases same : keyOf value = key <;> simp [same, Ne.symm]

theorem Partition.bucket_values {keyOf : String → Key} {items : List String} {buckets : List Bucket}
    (partition : Partition keyOf items buckets) (bucket : Bucket) (member : bucket ∈ buckets) :
    bucket.values = items.filter (fun value => decide (keyOf value = bucket.key)) := by
  have selected := find_of_mem buckets bucket member partition.unique
  simpa only [PatternBuckets.contents, selected, Option.map_some, Option.getD_some] using partition.contents bucket.key

/-- Every input value is present in the unique bucket selected by its key. -/
theorem Partition.member {keyOf : String → Key} {items : List String} {buckets : List Bucket}
    (partition : Partition keyOf items buckets) (value : String) (member : value ∈ items) :
    ∃ bucket ∈ buckets, bucket.key = keyOf value ∧ value ∈ bucket.values := by
  have present : value ∈ PatternBuckets.contents buckets (keyOf value) := by
    rw [partition.contents]
    simp [member]
  cases selected : find buckets (keyOf value) with
  | none => simp [PatternBuckets.contents, selected] at present
  | some bucket =>
      refine ⟨bucket, List.mem_of_find?_eq_some selected, ?_, ?_⟩
      · exact of_decide_eq_true
          (List.find?_some (p := fun entry : Bucket => decide (entry.key = keyOf value)) selected)
      · simpa only [PatternBuckets.contents, selected, Option.map_some, Option.getD_some] using present

structure View (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) : Prop where
  found : heap[dictionary]? = some (object buckets)
  lists : ∀ bucket ∈ buckets, heap[bucket.address]? = some (.list (texts bucket.values))
  afterDictionary : ∀ bucket ∈ buckets, dictionary < bucket.address
  unique : (buckets.map Bucket.address).Nodup

theorem View.append {heap : Mutable.Heap} {dictionary : Nat} {buckets : List Bucket}
    (view : View heap dictionary buckets) (extra : Mutable.Heap) : View (heap ++ extra) dictionary buckets :=
  ⟨HeapCollection.prefix_found heap extra dictionary _ view.found,
    fun bucket member => HeapCollection.prefix_found heap extra bucket.address _ (view.lists bucket member),
    view.afterDictionary, view.unique⟩

/-- The square-membership set cannot alias the dictionary or any bucket list. -/
theorem View.write_set {heap : Mutable.Heap} {dictionary : Nat} {buckets : List Bucket}
    (view : View heap dictionary buckets) (address : Nat) (old replacement : List Constant)
    (found : heap[address]? = some (.set old)) : View (heap.set address (.set replacement)) dictionary buckets := by
  have different : address ≠ dictionary := by
    intro same
    have original := view.found
    rw [← same, found] at original
    cases buckets <;> simp [object] at original
  refine ⟨?_, ?_, view.afterDictionary, view.unique⟩
  · rw [List.getElem?_set_ne different]
    exact view.found
  · intro bucket member
    have row := view.lists bucket member
    have separate : address ≠ bucket.address := by intro same; rw [← same, found] at row; cases row
    rw [List.getElem?_set_ne separate]
    exact row

theorem empty_view (heap : Mutable.Heap) : View (heap ++ [.dict []]) heap.length [] := by
  constructor
  · simp [object]
  · simp
  · simp
  · simp

def prepared (heap : Mutable.Heap) : Mutable.Heap := heap ++ [.list []]

def pending (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (key : Key) (representative : Nat) (value : String) : Mutable.Heap :=
  if (find buckets key).isSome then prepared heap
  else (prepared heap).set dictionary (object (push buckets key representative heap.length value))

/-- Lookup uses the existing pattern tuple and eagerly allocates the default list. -/
theorem setdefault (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (found : heap[dictionary]? = some (object buckets)) (key : Key) (representative : Nat) (value : String)
    (keyFound : heap[representative]? = some (.tuple (key.map number))) :
    Mutable.dictionarySetDefault dictionary (.ref representative) (.ref heap.length) (prepared heap) =
      some (.ok (.ref (chosen buckets key heap.length)), pending heap dictionary buckets key representative value) := by
  have original : (prepared heap)[dictionary]? = some (object buckets) :=
    HeapCollection.prefix_found heap _ dictionary _ found
  have hashed : Mutable.hashKey (.ref representative) (prepared heap) = some (.ok (hash key), prepared heap) := by
    have kept := HeapCollection.prefix_found heap [.list []] representative _ keyFound
    simpa [hash, number, List.map_map, Function.comp_def] using
      Mutable.hashKey_tuple_scalars (prepared heap) representative (key.map fun (n : Nat) => Constant.int n) (by simpa [prepared, number, List.map_map, Function.comp_def] using kept)
  cases buckets with
  | nil =>
      have operation := Mutable.dictionarySetDefault_promote (prepared heap) dictionary [] original
        representative (hash key) hashed (.ref heap.length)
      simpa [pending, find, chosen, object, push, entries, entry, Mutable.promoteDictionary] using operation
  | cons first rest =>
      have operation := Mutable.dictionarySetDefault_keyed (prepared heap) dictionary
        (entries (first :: rest)) original (.ref representative) (hash key) hashed (.ref heap.length)
      cases selected : find (first :: rest) key with
      | none =>
          simpa only [lookup_entries, selected, Option.map_none, Option.getD_none, Option.isSome_none,
            Bool.false_eq_true, if_false, pending, chosen, object_push, entries_push] using operation
      | some bucket =>
          simpa only [lookup_entries, selected, Option.map_some, Option.getD_some, Option.isSome_some,
            if_true, pending, chosen] using operation

theorem selected_values (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (representative : Nat) (value : String) :
    (pending heap dictionary buckets key representative value)[chosen buckets key heap.length]? =
      some (.list (texts (contents buckets key))) := by
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      have different : heap.length ≠ dictionary := by omega
      simp only [pending, chosen, contents, selected, Option.isSome_none, Bool.false_eq_true,
        if_false, Option.map_none, Option.getD_none]
      rw [List.getElem?_set_ne (Ne.symm different)]
      simp [prepared, texts]
  | some bucket =>
      have member : bucket ∈ buckets := List.mem_of_find?_eq_some selected
      have found := view.lists bucket member
      simpa [pending, chosen, contents, selected] using
        HeapCollection.prefix_found heap [.list []] bucket.address _ found

def updated (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (key : Key) (representative : Nat) (value : String) : Mutable.Heap :=
  (pending heap dictionary buckets key representative value).set (chosen buckets key heap.length)
    (.list (texts (contents buckets key ++ [value])))

theorem chosen_after (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) : dictionary < chosen buckets key heap.length := by
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      simp only [chosen, selected, Option.map_none, Option.getD_none]
      omega
  | some bucket =>
      simpa only [chosen, selected, Option.map_some, Option.getD_some] using
        view.afterDictionary bucket (List.mem_of_find?_eq_some selected)

theorem updated_selected (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (representative : Nat) (value : String) :
    (updated heap dictionary buckets key representative value)[chosen buckets key heap.length]? =
      some (.list (texts (contents buckets key ++ [value]))) := by
  exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp (selected_values heap dictionary buckets view key representative value)).1

theorem updated_other (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (representative : Nat) (value : String) (bucket : Bucket)
    (member : bucket ∈ buckets) (different : chosen buckets key heap.length ≠ bucket.address) :
    (updated heap dictionary buckets key representative value)[bucket.address]? = heap[bucket.address]? := by
  unfold updated
  rw [List.getElem?_set_ne different]
  have inside := (List.getElem?_eq_some_iff.mp (view.lists bucket member)).1
  have afterDictionary := view.afterDictionary bucket member
  unfold pending
  split
  · exact List.getElem?_append_left inside
  · rw [List.getElem?_set_ne (by omega : dictionary ≠ bucket.address)]
    exact List.getElem?_append_left inside

theorem updated_dictionary (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (representative : Nat) (value : String) :
    (updated heap dictionary buckets key representative value)[dictionary]? =
      some (object (push buckets key representative heap.length value)) := by
  have different := chosen_after heap dictionary buckets view key
  unfold updated
  rw [List.getElem?_set_ne (by omega : chosen buckets key heap.length ≠ dictionary)]
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      simp only [pending, selected, Option.isSome_none, Bool.false_eq_true, if_false]
      exact List.getElem?_set_self (by simp [prepared]; omega)
  | some bucket =>
      have same : object (push buckets key representative heap.length value) = object buckets := by
        rw [object_push, entries_push_hit buckets key _ _ _ bucket selected]
        cases buckets with
        | nil => simp [find] at selected
        | cons first rest => rfl
      rw [same]
      simpa only [pending, selected, Option.isSome_some, if_true] using
        HeapCollection.prefix_found heap _ dictionary _ view.found

/-- All writes stay at or above the dictionary, so a preceding input heap remains unchanged. -/
theorem updated_prefix (base suffix : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View (base ++ suffix) dictionary buckets) (afterBase : base.length ≤ dictionary)
    (key : Key) (representative : Nat) (value : String) :
    ∃ extra, updated (base ++ suffix) dictionary buckets key representative value = base ++ extra := by
  have afterChosen : base.length ≤ chosen buckets key (base ++ suffix).length := by
    have later := chosen_after (base ++ suffix) dictionary buckets view key
    omega
  have preparedPrefix : prepared (base ++ suffix) =
      base ++ (suffix ++ [.list []]) := by simp [prepared, List.append_assoc]
  unfold updated pending
  split
  · rw [preparedPrefix, List.set_append_right _ _ afterChosen]
    exact ⟨_, rfl⟩
  · rw [preparedPrefix, List.set_append_right _ _ afterBase, List.set_append_right _ _ afterChosen]
    exact ⟨_, rfl⟩

/-- Appending to one bucket preserves the dictionary, every other list, and absence of aliases. -/
theorem updated_view (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (representative : Nat) (value : String) :
    View (updated heap dictionary buckets key representative value) dictionary
      (push buckets key representative heap.length value) := by
  have fresh : heap.length ∉ buckets.map Bucket.address := by
    intro member
    obtain ⟨bucket, inside, equal⟩ := List.mem_map.mp member
    have bound := (List.getElem?_eq_some_iff.mp (view.lists bucket inside)).1
    omega
  constructor
  · exact updated_dictionary heap dictionary buckets view key representative value
  · intro current member
    cases selected : find buckets key with
    | none =>
        rw [push_miss buckets key _ _ _ selected] at member
        rcases List.mem_append.mp member with old | added
        · have bound := (List.getElem?_eq_some_iff.mp (view.lists current old)).1
          have different : chosen buckets key heap.length ≠ current.address := by
            simp only [chosen, selected, Option.map_none, Option.getD_none]
            omega
          rw [updated_other heap dictionary buckets view key representative value current old different]
          exact view.lists current old
        · obtain rfl := List.mem_singleton.mp added
          simpa only [chosen, contents, selected, Option.map_none, Option.getD_none, List.nil_append] using
            updated_selected heap dictionary buckets view key representative value
    | some bucket =>
        obtain ⟨matched, before, after, split, skipped⟩ := List.find?_eq_some_iff_append.mp selected
        have keyEqual : bucket.key = key := of_decide_eq_true matched
        have skip : ∀ entry ∈ before, entry.key ≠ key := by
          intro entry inside
          simpa using skipped entry inside
        have distinct : (before.map Bucket.address ++ bucket.address :: after.map Bucket.address).Nodup := by
          simpa only [split, List.map_append, List.map_cons] using view.unique
        have moved : (bucket.address :: (before.map Bucket.address ++ after.map Bucket.address)).Nodup :=
          List.perm_middle.nodup_iff.mp distinct
        have without := (List.nodup_cons.mp moved).1
        have kept (entry : Bucket) (inside : entry ∈ before ++ after) :
            (updated heap dictionary buckets key representative value)[entry.address]? = some (.list (texts entry.values)) := by
          have old : entry ∈ buckets := by
            rw [split]
            rcases List.mem_append.mp inside with left | right
            · exact List.mem_append.mpr (.inl left)
            · exact List.mem_append.mpr (.inr (List.mem_cons_of_mem _ right))
          have different : chosen buckets key heap.length ≠ entry.address := by
            simp only [chosen, selected, Option.map_some, Option.getD_some]
            intro equal
            apply without
            rw [equal]
            simpa only [List.map_append] using List.mem_map_of_mem (f := Bucket.address) inside
          rw [updated_other heap dictionary buckets view key representative value entry old different]
          exact view.lists entry old
        rw [split, push_split before after bucket key keyEqual skip] at member
        rcases List.mem_append.mp member with left | right
        · exact kept current (List.mem_append.mpr (.inl left))
        · rcases List.mem_cons.mp right with rfl | right
          · simpa only [chosen, contents, selected, Option.map_some, Option.getD_some] using
              updated_selected heap dictionary buckets view key representative value
          · exact kept current (List.mem_append.mpr (.inr right))
  · intro current member
    have addressMember := List.mem_map_of_mem (f := Bucket.address) member
    rw [addresses_push] at addressMember
    cases selected : find buckets key with
    | none =>
        simp only [selected, Option.isSome_none, Bool.false_eq_true, if_false] at addressMember
        rcases List.mem_append.mp addressMember with old | added
        · obtain ⟨entry, inside, equal⟩ := List.mem_map.mp old
          rw [← equal]
          exact view.afterDictionary entry inside
        · have equal := List.mem_singleton.mp added
          have bound := (List.getElem?_eq_some_iff.mp view.found).1
          omega
    | some bucket =>
        simp only [selected, Option.isSome_some, if_true] at addressMember
        obtain ⟨entry, inside, equal⟩ := List.mem_map.mp addressMember
        rw [← equal]
        exact view.afterDictionary entry inside
  · rw [addresses_push]
    split
    · exact view.unique
    · simpa only [List.concat_eq_append] using List.Nodup.concat fresh view.unique

/-- Appending a square string leaves the independently stored membership set unchanged. -/
theorem updated_set (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : Key) (representative : Nat) (value : String)
    (address : Nat) (items : List Constant) (found : heap[address]? = some (.set items)) :
    (updated heap dictionary buckets key representative value)[address]? = some (.set items) := by
  have dictionaryDifferent : dictionary ≠ address := by
    intro same
    have original := view.found
    rw [same, found] at original
    cases buckets <;> simp [object] at original
  have chosenDifferent : chosen buckets key heap.length ≠ address := by
    cases selected : find buckets key with
    | none =>
        have bound := (List.getElem?_eq_some_iff.mp found).1
        simp only [chosen, selected, Option.map_none, Option.getD_none]
        omega
    | some bucket =>
        have row := view.lists bucket (List.mem_of_find?_eq_some selected)
        simp only [chosen, selected, Option.map_some, Option.getD_some]
        intro same
        rw [same, found] at row
        cases row
  unfold updated
  rw [List.getElem?_set_ne chosenDifferent]
  unfold pending
  split
  · exact HeapCollection.prefix_found heap [.list []] address _ found
  · rw [List.getElem?_set_ne dictionaryDifferent]
    exact HeapCollection.prefix_found heap [.list []] address _ found

def insertion : Expr := .call (.attribute (.call (.attribute (.name "by_pat") "setdefault")
  [.name "pat", .list []]) "append") [.name "s"]

/-- Execute setdefault and append using the actual tuple reference returned by pattern_signature. -/
theorem evaluate {globals : Mutable.Globals} (program : Module) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (view : View heap dictionary buckets)
    (key : Key) (representative : Nat) (value : String)
    (getMap : env "by_pat" = some (.ref dictionary)) (getKey : env "pat" = some (.ref representative))
    (getValue : env "s" = some (text value)) (keyFound : heap[representative]? = some (.tuple (key.map number))) :
    Evaluates program env insertion heap (.scalar .none) (updated heap dictionary buckets key representative value) globals := by
  have bound : bindMethod program (.ref dictionary) "setdefault" heap =
      some (.ok (.dictionarySetDefault dictionary), heap) := by
    have found := view.found
    cases buckets <;> simp_all [bindMethod, object]
  have receiver : Evaluates program env (.call (.attribute (.name "by_pat") "setdefault") [.name "pat", .list []])
      heap (.ref (chosen buckets key heap.length)) (pending heap dictionary buckets key representative value) globals := by
    apply Evaluates.dictionary_setdefault (by intro name same; cases same; simp [getMap]) (Evaluates.lookup getMap) bound
      (.cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil))
    exact setdefault heap dictionary buckets view.found key representative value keyFound
  have selected := selected_values heap dictionary buckets view key representative value
  simpa only [insertion, updated, texts, List.map_append, List.map_cons, List.map_nil] using
    Evaluates.append receiver selected (Evaluates.lookup getValue) selected

end ProjectEulerSolutions.PatternBuckets
