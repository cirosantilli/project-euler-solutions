import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9679_small : List Nat := []
theorem row9679_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9679 row9679_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9679_block_005 : List Nat := [10093]
def plan9679_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 17), .first, .second, (.firstFactor 37), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row9679_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9679 row9679_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9679_block_005
  · decide
  · rfl

def row9679_block_006 : List Nat := [10687, 11059, 11383, 11719, 11863]
def plan9679_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .possible, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 29), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .possible, .second, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9679_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9679 row9679_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9679_block_006
  · decide
  · rfl

def row9679_block_007 : List Nat := [12973, 13711]
def plan9679_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 31), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .second, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 23), .first, .first, (.secondFactor 31), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row9679_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9679 row9679_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9679_block_007
  · decide
  · rfl

def row9679_block_008 : List Nat := [15217, 15787, 15913]
def plan9679_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 13), .possible, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .possible, (.firstFactor 29), .first, (.secondFactor 29), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9679_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9679 row9679_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9679_block_008
  · decide
  · rfl

def row9679_block_009 : List Nat := []
theorem row9679_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9679 row9679_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9679_block_010 : List Nat := []
theorem row9679_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9679 row9679_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9679_block_011 : List Nat := []
theorem row9679_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9679 row9679_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9679_block_012 : List Nat := []
theorem row9679_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9679 row9679_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9679_block_013 : List Nat := []
theorem row9679_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9679 row9679_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9679_large_left_left : List Nat := row9679_small ++ row9679_block_005
theorem row9679_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9679 row9679_large_left_left :=
  Budget.RowUpperBound.branch row9679_small_covers row9679_block_005_covers

def row9679_large_left_right_right : List Nat := row9679_block_007 ++ row9679_block_008
theorem row9679_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9679 row9679_large_left_right_right :=
  Budget.RowUpperBound.branch row9679_block_007_covers row9679_block_008_covers

def row9679_large_left_right : List Nat := row9679_block_006 ++ row9679_large_left_right_right
theorem row9679_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9679 row9679_large_left_right :=
  Budget.RowUpperBound.branch row9679_block_006_covers row9679_large_left_right_right_covers

def row9679_large_left : List Nat := row9679_large_left_left ++ row9679_large_left_right
theorem row9679_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9679 row9679_large_left :=
  Budget.RowUpperBound.branch row9679_large_left_left_covers row9679_large_left_right_covers

def row9679_large_right_left : List Nat := row9679_block_009 ++ row9679_block_010
theorem row9679_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9679 row9679_large_right_left :=
  Budget.RowUpperBound.branch row9679_block_009_covers row9679_block_010_covers

def row9679_large_right_right_right : List Nat := row9679_block_012 ++ row9679_block_013
theorem row9679_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9679 row9679_large_right_right_right :=
  Budget.RowUpperBound.branch row9679_block_012_covers row9679_block_013_covers

def row9679_large_right_right : List Nat := row9679_block_011 ++ row9679_large_right_right_right
theorem row9679_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9679 row9679_large_right_right :=
  Budget.RowUpperBound.branch row9679_block_011_covers row9679_large_right_right_right_covers

def row9679_large_right : List Nat := row9679_large_right_left ++ row9679_large_right_right
theorem row9679_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9679 row9679_large_right :=
  Budget.RowUpperBound.branch row9679_large_right_left_covers row9679_large_right_right_covers

def row9679_large : List Nat := row9679_large_left ++ row9679_large_right
theorem row9679_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9679 row9679_large :=
  Budget.RowUpperBound.branch row9679_large_left_covers row9679_large_right_covers

def row9679 : List Nat := row9679_large
theorem row9679_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9679 row9679 := row9679_large_covers

def row9689_small : List Nat := []
theorem row9689_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9689 row9689_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9689_block_005 : List Nat := [9743, 9923]
def plan9689_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, .first, .possible, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 19), .second, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9689_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9689 row9689_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9689_block_005
  · decide
  · rfl

def row9689_block_006 : List Nat := [11969, 12281]
def plan9689_block_006 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .second, (.firstFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, (.secondFactor 29), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9689_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9689 row9689_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9689_block_006
  · decide
  · rfl

def row9689_block_007 : List Nat := [13469, 14087]
def plan9689_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 29), .second, .first, (.secondFactor 19), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, (.firstFactor 17), (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row9689_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9689 row9689_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9689_block_007
  · decide
  · rfl

def row9689_block_008 : List Nat := [14879, 14897, 15749, 15791]
def plan9689_block_008 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 19), (.secondFactor 11), .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 31), .first, .possible, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 31), .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 11), (.secondFactor 19), (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9689_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9689 row9689_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9689_block_008
  · decide
  · rfl

def row9689_block_009 : List Nat := []
theorem row9689_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9689 row9689_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9689_block_010 : List Nat := []
theorem row9689_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9689 row9689_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9689_block_011 : List Nat := []
theorem row9689_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9689 row9689_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9689_block_012 : List Nat := []
theorem row9689_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9689 row9689_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9689_block_013 : List Nat := []
theorem row9689_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9689 row9689_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9689_large_left_left : List Nat := row9689_small ++ row9689_block_005
theorem row9689_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9689 row9689_large_left_left :=
  Budget.RowUpperBound.branch row9689_small_covers row9689_block_005_covers

def row9689_large_left_right_right : List Nat := row9689_block_007 ++ row9689_block_008
theorem row9689_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9689 row9689_large_left_right_right :=
  Budget.RowUpperBound.branch row9689_block_007_covers row9689_block_008_covers

def row9689_large_left_right : List Nat := row9689_block_006 ++ row9689_large_left_right_right
theorem row9689_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9689 row9689_large_left_right :=
  Budget.RowUpperBound.branch row9689_block_006_covers row9689_large_left_right_right_covers

def row9689_large_left : List Nat := row9689_large_left_left ++ row9689_large_left_right
theorem row9689_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9689 row9689_large_left :=
  Budget.RowUpperBound.branch row9689_large_left_left_covers row9689_large_left_right_covers

def row9689_large_right_left : List Nat := row9689_block_009 ++ row9689_block_010
theorem row9689_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9689 row9689_large_right_left :=
  Budget.RowUpperBound.branch row9689_block_009_covers row9689_block_010_covers

def row9689_large_right_right_right : List Nat := row9689_block_012 ++ row9689_block_013
theorem row9689_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9689 row9689_large_right_right_right :=
  Budget.RowUpperBound.branch row9689_block_012_covers row9689_block_013_covers

def row9689_large_right_right : List Nat := row9689_block_011 ++ row9689_large_right_right_right
theorem row9689_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9689 row9689_large_right_right :=
  Budget.RowUpperBound.branch row9689_block_011_covers row9689_large_right_right_right_covers

def row9689_large_right : List Nat := row9689_large_right_left ++ row9689_large_right_right
theorem row9689_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9689 row9689_large_right :=
  Budget.RowUpperBound.branch row9689_large_right_left_covers row9689_large_right_right_covers

def row9689_large : List Nat := row9689_large_left ++ row9689_large_right
theorem row9689_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9689 row9689_large :=
  Budget.RowUpperBound.branch row9689_large_left_covers row9689_large_right_covers

def row9689 : List Nat := row9689_large
theorem row9689_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9689 row9689 := row9689_large_covers

def row9697_small : List Nat := []
theorem row9697_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9697 row9697_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9697_block_005 : List Nat := [10333, 10369]
def plan9697_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .possible]
set_option maxRecDepth 2048 in
theorem row9697_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9697 row9697_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9697_block_005
  · decide
  · rfl

def row9697_block_006 : List Nat := [10453, 10459, 11923, 12097, 12277, 12343]
def plan9697_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 19), .first, .possible, .first, .possible, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, .second, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 11), .first, .second, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 11), .possible, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row9697_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9697 row9697_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9697_block_006
  · decide
  · rfl

def row9697_block_007 : List Nat := [13309, 14071]
def plan9697_block_007 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 29), .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 31), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 31), (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row9697_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9697 row9697_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9697_block_007
  · decide
  · rfl

def row9697_block_008 : List Nat := [14593, 14929]
def plan9697_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9697_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9697 row9697_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9697_block_008
  · decide
  · rfl

def row9697_block_009 : List Nat := []
theorem row9697_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9697 row9697_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9697_block_010 : List Nat := []
theorem row9697_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9697 row9697_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9697_block_011 : List Nat := []
theorem row9697_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9697 row9697_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9697_block_012 : List Nat := []
theorem row9697_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9697 row9697_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9697_block_013 : List Nat := []
theorem row9697_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9697 row9697_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9697_large_left_left : List Nat := row9697_small ++ row9697_block_005
theorem row9697_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9697 row9697_large_left_left :=
  Budget.RowUpperBound.branch row9697_small_covers row9697_block_005_covers

def row9697_large_left_right_right : List Nat := row9697_block_007 ++ row9697_block_008
theorem row9697_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9697 row9697_large_left_right_right :=
  Budget.RowUpperBound.branch row9697_block_007_covers row9697_block_008_covers

def row9697_large_left_right : List Nat := row9697_block_006 ++ row9697_large_left_right_right
theorem row9697_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9697 row9697_large_left_right :=
  Budget.RowUpperBound.branch row9697_block_006_covers row9697_large_left_right_right_covers

def row9697_large_left : List Nat := row9697_large_left_left ++ row9697_large_left_right
theorem row9697_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9697 row9697_large_left :=
  Budget.RowUpperBound.branch row9697_large_left_left_covers row9697_large_left_right_covers

def row9697_large_right_left : List Nat := row9697_block_009 ++ row9697_block_010
theorem row9697_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9697 row9697_large_right_left :=
  Budget.RowUpperBound.branch row9697_block_009_covers row9697_block_010_covers

def row9697_large_right_right_right : List Nat := row9697_block_012 ++ row9697_block_013
theorem row9697_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9697 row9697_large_right_right_right :=
  Budget.RowUpperBound.branch row9697_block_012_covers row9697_block_013_covers

def row9697_large_right_right : List Nat := row9697_block_011 ++ row9697_large_right_right_right
theorem row9697_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9697 row9697_large_right_right :=
  Budget.RowUpperBound.branch row9697_block_011_covers row9697_large_right_right_right_covers

def row9697_large_right : List Nat := row9697_large_right_left ++ row9697_large_right_right
theorem row9697_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9697 row9697_large_right :=
  Budget.RowUpperBound.branch row9697_large_right_left_covers row9697_large_right_right_covers

def row9697_large : List Nat := row9697_large_left ++ row9697_large_right
theorem row9697_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9697 row9697_large :=
  Budget.RowUpperBound.branch row9697_large_left_covers row9697_large_right_covers

def row9697 : List Nat := row9697_large
theorem row9697_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9697 row9697 := row9697_large_covers

def row9719_small : List Nat := []
theorem row9719_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9719 row9719_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9719_block_005 : List Nat := [10223]
def plan9719_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 19), .possible, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9719_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9719 row9719_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9719_block_005
  · decide
  · rfl

def row9719_block_006 : List Nat := [10427, 10691, 10889, 11813, 11939]
def plan9719_block_006 : List Budget.Choice := [(.secondFactor 31), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 17), .first, .first, .first, .possible, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9719_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9719 row9719_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9719_block_006
  · decide
  · rfl

def row9719_block_007 : List Nat := [13037, 13043, 14249]
def plan9719_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .second, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .possible, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 29), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 31), (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 23), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9719_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9719 row9719_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9719_block_007
  · decide
  · rfl

def row9719_block_008 : List Nat := [15731]
def plan9719_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .second, (.secondFactor 17), (.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, (.secondFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .second, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9719_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9719 row9719_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9719_block_008
  · decide
  · rfl

def row9719_block_009 : List Nat := []
theorem row9719_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9719 row9719_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9719_block_010 : List Nat := []
theorem row9719_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9719 row9719_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9719_block_011 : List Nat := []
theorem row9719_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9719 row9719_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9719_block_012 : List Nat := []
theorem row9719_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9719 row9719_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9719_block_013 : List Nat := []
theorem row9719_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9719 row9719_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9719_large_left_left : List Nat := row9719_small ++ row9719_block_005
theorem row9719_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9719 row9719_large_left_left :=
  Budget.RowUpperBound.branch row9719_small_covers row9719_block_005_covers

def row9719_large_left_right_right : List Nat := row9719_block_007 ++ row9719_block_008
theorem row9719_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9719 row9719_large_left_right_right :=
  Budget.RowUpperBound.branch row9719_block_007_covers row9719_block_008_covers

def row9719_large_left_right : List Nat := row9719_block_006 ++ row9719_large_left_right_right
theorem row9719_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9719 row9719_large_left_right :=
  Budget.RowUpperBound.branch row9719_block_006_covers row9719_large_left_right_right_covers

def row9719_large_left : List Nat := row9719_large_left_left ++ row9719_large_left_right
theorem row9719_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9719 row9719_large_left :=
  Budget.RowUpperBound.branch row9719_large_left_left_covers row9719_large_left_right_covers

def row9719_large_right_left : List Nat := row9719_block_009 ++ row9719_block_010
theorem row9719_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9719 row9719_large_right_left :=
  Budget.RowUpperBound.branch row9719_block_009_covers row9719_block_010_covers

def row9719_large_right_right_right : List Nat := row9719_block_012 ++ row9719_block_013
theorem row9719_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9719 row9719_large_right_right_right :=
  Budget.RowUpperBound.branch row9719_block_012_covers row9719_block_013_covers

def row9719_large_right_right : List Nat := row9719_block_011 ++ row9719_large_right_right_right
theorem row9719_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9719 row9719_large_right_right :=
  Budget.RowUpperBound.branch row9719_block_011_covers row9719_large_right_right_right_covers

def row9719_large_right : List Nat := row9719_large_right_left ++ row9719_large_right_right
theorem row9719_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9719 row9719_large_right :=
  Budget.RowUpperBound.branch row9719_large_right_left_covers row9719_large_right_right_covers

def row9719_large : List Nat := row9719_large_left ++ row9719_large_right
theorem row9719_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9719 row9719_large :=
  Budget.RowUpperBound.branch row9719_large_left_covers row9719_large_right_covers

def row9719 : List Nat := row9719_large
theorem row9719_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9719 row9719 := row9719_large_covers

end ProjectEulerSolutions.PrimePairLarge
