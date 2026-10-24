import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9749_small : List Nat := []
theorem row9749_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9749 row9749_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9749_block_005 : List Nat := []
def plan9749_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9749_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9749 row9749_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9749_block_005
  · decide
  · rfl

def row9749_block_006 : List Nat := [11621, 12161]
def plan9749_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .second, .first, .second, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .possible, .second, (.secondFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .second, (.firstFactor 31), .first, .possible, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9749_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9749 row9749_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9749_block_006
  · decide
  · rfl

def row9749_block_007 : List Nat := [13757, 14243]
def plan9749_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .second, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 13), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9749_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9749 row9749_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9749_block_007
  · decide
  · rfl

def row9749_block_008 : List Nat := [14783, 14951, 16073, 16193]
def plan9749_block_008 : List Budget.Choice := [.first, (.firstFactor 29), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .second, (.secondFactor 23), (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 17), .first, .possible, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .possible, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9749_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9749 row9749_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9749_block_008
  · decide
  · rfl

def row9749_block_009 : List Nat := []
theorem row9749_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9749 row9749_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9749_block_010 : List Nat := []
theorem row9749_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9749 row9749_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9749_block_011 : List Nat := []
theorem row9749_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9749 row9749_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9749_block_012 : List Nat := []
theorem row9749_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9749 row9749_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9749_block_013 : List Nat := []
theorem row9749_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9749 row9749_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9749_large_left_left : List Nat := row9749_small ++ row9749_block_005
theorem row9749_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9749 row9749_large_left_left :=
  Budget.RowUpperBound.branch row9749_small_covers row9749_block_005_covers

def row9749_large_left_right_right : List Nat := row9749_block_007 ++ row9749_block_008
theorem row9749_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9749 row9749_large_left_right_right :=
  Budget.RowUpperBound.branch row9749_block_007_covers row9749_block_008_covers

def row9749_large_left_right : List Nat := row9749_block_006 ++ row9749_large_left_right_right
theorem row9749_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9749 row9749_large_left_right :=
  Budget.RowUpperBound.branch row9749_block_006_covers row9749_large_left_right_right_covers

def row9749_large_left : List Nat := row9749_large_left_left ++ row9749_large_left_right
theorem row9749_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9749 row9749_large_left :=
  Budget.RowUpperBound.branch row9749_large_left_left_covers row9749_large_left_right_covers

def row9749_large_right_left : List Nat := row9749_block_009 ++ row9749_block_010
theorem row9749_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9749 row9749_large_right_left :=
  Budget.RowUpperBound.branch row9749_block_009_covers row9749_block_010_covers

def row9749_large_right_right_right : List Nat := row9749_block_012 ++ row9749_block_013
theorem row9749_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9749 row9749_large_right_right_right :=
  Budget.RowUpperBound.branch row9749_block_012_covers row9749_block_013_covers

def row9749_large_right_right : List Nat := row9749_block_011 ++ row9749_large_right_right_right
theorem row9749_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9749 row9749_large_right_right :=
  Budget.RowUpperBound.branch row9749_block_011_covers row9749_large_right_right_right_covers

def row9749_large_right : List Nat := row9749_large_right_left ++ row9749_large_right_right
theorem row9749_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9749 row9749_large_right :=
  Budget.RowUpperBound.branch row9749_large_right_left_covers row9749_large_right_right_covers

def row9749_large : List Nat := row9749_large_left ++ row9749_large_right
theorem row9749_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9749 row9749_large :=
  Budget.RowUpperBound.branch row9749_large_left_covers row9749_large_right_covers

def row9749 : List Nat := row9749_large
theorem row9749_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9749 row9749 := row9749_large_covers

def row9767_small : List Nat := []
theorem row9767_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9767 row9767_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9767_block_005 : List Nat := [9833]
def plan9767_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 31), (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9767_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9767 row9767_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9767_block_005
  · decide
  · rfl

def row9767_block_006 : List Nat := [10463, 10613, 11867, 12329]
def plan9767_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, .possible, .first, .first, .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, .second, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .second, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9767_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9767 row9767_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9767_block_006
  · decide
  · rfl

def row9767_block_007 : List Nat := [12983, 13763]
def plan9767_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 29), (.secondFactor 11), .first, .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 37), .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row9767_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9767 row9767_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9767_block_007
  · decide
  · rfl

def row9767_block_008 : List Nat := [14561, 14891, 15017, 15923]
def plan9767_block_008 : List Budget.Choice := [.first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, .second, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 19), (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .second, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 11), .first, .second, (.firstFactor 17), (.firstFactor 19), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9767_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9767 row9767_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9767_block_008
  · decide
  · rfl

def row9767_block_009 : List Nat := []
theorem row9767_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9767 row9767_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9767_block_010 : List Nat := []
theorem row9767_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9767 row9767_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9767_block_011 : List Nat := []
theorem row9767_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9767 row9767_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9767_block_012 : List Nat := []
theorem row9767_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9767 row9767_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9767_block_013 : List Nat := []
theorem row9767_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9767 row9767_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9767_large_left_left : List Nat := row9767_small ++ row9767_block_005
theorem row9767_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9767 row9767_large_left_left :=
  Budget.RowUpperBound.branch row9767_small_covers row9767_block_005_covers

def row9767_large_left_right_right : List Nat := row9767_block_007 ++ row9767_block_008
theorem row9767_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9767 row9767_large_left_right_right :=
  Budget.RowUpperBound.branch row9767_block_007_covers row9767_block_008_covers

def row9767_large_left_right : List Nat := row9767_block_006 ++ row9767_large_left_right_right
theorem row9767_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9767 row9767_large_left_right :=
  Budget.RowUpperBound.branch row9767_block_006_covers row9767_large_left_right_right_covers

def row9767_large_left : List Nat := row9767_large_left_left ++ row9767_large_left_right
theorem row9767_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9767 row9767_large_left :=
  Budget.RowUpperBound.branch row9767_large_left_left_covers row9767_large_left_right_covers

def row9767_large_right_left : List Nat := row9767_block_009 ++ row9767_block_010
theorem row9767_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9767 row9767_large_right_left :=
  Budget.RowUpperBound.branch row9767_block_009_covers row9767_block_010_covers

def row9767_large_right_right_right : List Nat := row9767_block_012 ++ row9767_block_013
theorem row9767_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9767 row9767_large_right_right_right :=
  Budget.RowUpperBound.branch row9767_block_012_covers row9767_block_013_covers

def row9767_large_right_right : List Nat := row9767_block_011 ++ row9767_large_right_right_right
theorem row9767_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9767 row9767_large_right_right :=
  Budget.RowUpperBound.branch row9767_block_011_covers row9767_large_right_right_right_covers

def row9767_large_right : List Nat := row9767_large_right_left ++ row9767_large_right_right
theorem row9767_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9767 row9767_large_right :=
  Budget.RowUpperBound.branch row9767_large_right_left_covers row9767_large_right_right_covers

def row9767_large : List Nat := row9767_large_left ++ row9767_large_right
theorem row9767_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9767 row9767_large :=
  Budget.RowUpperBound.branch row9767_large_left_covers row9767_large_right_covers

def row9767 : List Nat := row9767_large
theorem row9767_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9767 row9767 := row9767_large_covers

def row9769_small : List Nat := []
theorem row9769_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9769 row9769_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9769_block_005 : List Nat := [9907, 10069]
def plan9769_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .second, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 23), .first, .first, .first, .first, .second, .second, (.firstFactor 23), .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9769_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9769 row9769_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9769_block_005
  · decide
  · rfl

def row9769_block_006 : List Nat := [10453, 11161]
def plan9769_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .second, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), (.secondFactor 29), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9769_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9769 row9769_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9769_block_006
  · decide
  · rfl

def row9769_block_007 : List Nat := [12421, 13759]
def plan9769_block_007 : List Budget.Choice := [.possible, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .second, .first, .second, (.secondFactor 11), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 37), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9769_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9769 row9769_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9769_block_007
  · decide
  · rfl

def row9769_block_008 : List Nat := [14929, 15391]
def plan9769_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .second, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9769_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9769 row9769_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9769_block_008
  · decide
  · rfl

def row9769_block_009 : List Nat := []
theorem row9769_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9769 row9769_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9769_block_010 : List Nat := []
theorem row9769_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9769 row9769_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9769_block_011 : List Nat := []
theorem row9769_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9769 row9769_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9769_block_012 : List Nat := []
theorem row9769_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9769 row9769_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9769_block_013 : List Nat := []
theorem row9769_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9769 row9769_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9769_large_left_left : List Nat := row9769_small ++ row9769_block_005
theorem row9769_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9769 row9769_large_left_left :=
  Budget.RowUpperBound.branch row9769_small_covers row9769_block_005_covers

def row9769_large_left_right_right : List Nat := row9769_block_007 ++ row9769_block_008
theorem row9769_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9769 row9769_large_left_right_right :=
  Budget.RowUpperBound.branch row9769_block_007_covers row9769_block_008_covers

def row9769_large_left_right : List Nat := row9769_block_006 ++ row9769_large_left_right_right
theorem row9769_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9769 row9769_large_left_right :=
  Budget.RowUpperBound.branch row9769_block_006_covers row9769_large_left_right_right_covers

def row9769_large_left : List Nat := row9769_large_left_left ++ row9769_large_left_right
theorem row9769_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9769 row9769_large_left :=
  Budget.RowUpperBound.branch row9769_large_left_left_covers row9769_large_left_right_covers

def row9769_large_right_left : List Nat := row9769_block_009 ++ row9769_block_010
theorem row9769_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9769 row9769_large_right_left :=
  Budget.RowUpperBound.branch row9769_block_009_covers row9769_block_010_covers

def row9769_large_right_right_right : List Nat := row9769_block_012 ++ row9769_block_013
theorem row9769_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9769 row9769_large_right_right_right :=
  Budget.RowUpperBound.branch row9769_block_012_covers row9769_block_013_covers

def row9769_large_right_right : List Nat := row9769_block_011 ++ row9769_large_right_right_right
theorem row9769_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9769 row9769_large_right_right :=
  Budget.RowUpperBound.branch row9769_block_011_covers row9769_large_right_right_right_covers

def row9769_large_right : List Nat := row9769_large_right_left ++ row9769_large_right_right
theorem row9769_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9769 row9769_large_right :=
  Budget.RowUpperBound.branch row9769_large_right_left_covers row9769_large_right_right_covers

def row9769_large : List Nat := row9769_large_left ++ row9769_large_right
theorem row9769_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9769 row9769_large :=
  Budget.RowUpperBound.branch row9769_large_left_covers row9769_large_right_covers

def row9769 : List Nat := row9769_large
theorem row9769_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9769 row9769 := row9769_large_covers

def row9781_small : List Nat := []
theorem row9781_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9781 row9781_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9781_block_005 : List Nat := [10009, 10369]
def plan9781_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, .first, (.secondFactor 19), (.secondFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .possible, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .possible]
set_option maxRecDepth 2048 in
theorem row9781_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9781 row9781_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9781_block_005
  · decide
  · rfl

def row9781_block_006 : List Nat := [11443, 11527, 12049]
def plan9781_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.secondFactor 11), .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 11), .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 29), .first, .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9781_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9781 row9781_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9781_block_006
  · decide
  · rfl

def row9781_block_007 : List Nat := [12781, 12907, 13399, 14197]
def plan9781_block_007 : List Budget.Choice := [(.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 37), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 31), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row9781_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9781 row9781_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9781_block_007
  · decide
  · rfl

def row9781_block_008 : List Nat := []
def plan9781_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 29), (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.secondFactor 29), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 29), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9781_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9781 row9781_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9781_block_008
  · decide
  · rfl

def row9781_block_009 : List Nat := []
theorem row9781_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9781 row9781_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9781_block_010 : List Nat := []
theorem row9781_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9781 row9781_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9781_block_011 : List Nat := []
theorem row9781_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9781 row9781_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9781_block_012 : List Nat := []
theorem row9781_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9781 row9781_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9781_block_013 : List Nat := []
theorem row9781_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9781 row9781_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9781_large_left_left : List Nat := row9781_small ++ row9781_block_005
theorem row9781_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9781 row9781_large_left_left :=
  Budget.RowUpperBound.branch row9781_small_covers row9781_block_005_covers

def row9781_large_left_right_right : List Nat := row9781_block_007 ++ row9781_block_008
theorem row9781_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9781 row9781_large_left_right_right :=
  Budget.RowUpperBound.branch row9781_block_007_covers row9781_block_008_covers

def row9781_large_left_right : List Nat := row9781_block_006 ++ row9781_large_left_right_right
theorem row9781_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9781 row9781_large_left_right :=
  Budget.RowUpperBound.branch row9781_block_006_covers row9781_large_left_right_right_covers

def row9781_large_left : List Nat := row9781_large_left_left ++ row9781_large_left_right
theorem row9781_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9781 row9781_large_left :=
  Budget.RowUpperBound.branch row9781_large_left_left_covers row9781_large_left_right_covers

def row9781_large_right_left : List Nat := row9781_block_009 ++ row9781_block_010
theorem row9781_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9781 row9781_large_right_left :=
  Budget.RowUpperBound.branch row9781_block_009_covers row9781_block_010_covers

def row9781_large_right_right_right : List Nat := row9781_block_012 ++ row9781_block_013
theorem row9781_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9781 row9781_large_right_right_right :=
  Budget.RowUpperBound.branch row9781_block_012_covers row9781_block_013_covers

def row9781_large_right_right : List Nat := row9781_block_011 ++ row9781_large_right_right_right
theorem row9781_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9781 row9781_large_right_right :=
  Budget.RowUpperBound.branch row9781_block_011_covers row9781_large_right_right_right_covers

def row9781_large_right : List Nat := row9781_large_right_left ++ row9781_large_right_right
theorem row9781_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9781 row9781_large_right :=
  Budget.RowUpperBound.branch row9781_large_right_left_covers row9781_large_right_right_covers

def row9781_large : List Nat := row9781_large_left ++ row9781_large_right
theorem row9781_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9781 row9781_large :=
  Budget.RowUpperBound.branch row9781_large_left_covers row9781_large_right_covers

def row9781 : List Nat := row9781_large
theorem row9781_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9781 row9781 := row9781_large_covers

end ProjectEulerSolutions.PrimePairLarge
