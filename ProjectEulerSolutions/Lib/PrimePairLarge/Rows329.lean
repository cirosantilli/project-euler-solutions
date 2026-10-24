import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10853_small : List Nat := []
theorem row10853_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10853 row10853_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10853_block_005 : List Nat := []
theorem row10853_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10853 row10853_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10853_block_006 : List Nat := [10937, 11897, 12251]
def plan10853_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 19), .first, (.firstFactor 37), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 31), (.firstFactor 17), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 7), (.firstFactor 17), (.secondFactor 29), .first, .first, (.secondFactor 37), .first, .first, .first, .first, .first, .possible, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 29), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row10853_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10853 row10853_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10853_block_006
  · decide
  · rfl

def row10853_block_007 : List Nat := [13367, 14207]
def plan10853_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 37), (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 19), (.firstFactor 13), .second, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 13), (.secondFactor 29), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10853_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10853 row10853_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10853_block_007
  · decide
  · rfl

def row10853_block_008 : List Nat := [15107, 15137]
def plan10853_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, (.secondFactor 37), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .possible, .first, (.firstFactor 19), .possible, .first, .first, (.firstFactor 7), (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10853_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10853 row10853_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10853_block_008
  · decide
  · rfl

def row10853_block_009 : List Nat := []
theorem row10853_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10853 row10853_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10853_block_010 : List Nat := []
theorem row10853_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10853 row10853_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10853_block_011 : List Nat := []
theorem row10853_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10853 row10853_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10853_block_012 : List Nat := []
theorem row10853_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10853 row10853_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10853_block_013 : List Nat := []
theorem row10853_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10853 row10853_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10853_large_left_left : List Nat := row10853_small ++ row10853_block_005
theorem row10853_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10853 row10853_large_left_left :=
  Budget.RowUpperBound.branch row10853_small_covers row10853_block_005_covers

def row10853_large_left_right_right : List Nat := row10853_block_007 ++ row10853_block_008
theorem row10853_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10853 row10853_large_left_right_right :=
  Budget.RowUpperBound.branch row10853_block_007_covers row10853_block_008_covers

def row10853_large_left_right : List Nat := row10853_block_006 ++ row10853_large_left_right_right
theorem row10853_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10853 row10853_large_left_right :=
  Budget.RowUpperBound.branch row10853_block_006_covers row10853_large_left_right_right_covers

def row10853_large_left : List Nat := row10853_large_left_left ++ row10853_large_left_right
theorem row10853_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10853 row10853_large_left :=
  Budget.RowUpperBound.branch row10853_large_left_left_covers row10853_large_left_right_covers

def row10853_large_right_left : List Nat := row10853_block_009 ++ row10853_block_010
theorem row10853_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10853 row10853_large_right_left :=
  Budget.RowUpperBound.branch row10853_block_009_covers row10853_block_010_covers

def row10853_large_right_right_right : List Nat := row10853_block_012 ++ row10853_block_013
theorem row10853_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10853 row10853_large_right_right_right :=
  Budget.RowUpperBound.branch row10853_block_012_covers row10853_block_013_covers

def row10853_large_right_right : List Nat := row10853_block_011 ++ row10853_large_right_right_right
theorem row10853_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10853 row10853_large_right_right :=
  Budget.RowUpperBound.branch row10853_block_011_covers row10853_large_right_right_right_covers

def row10853_large_right : List Nat := row10853_large_right_left ++ row10853_large_right_right
theorem row10853_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10853 row10853_large_right :=
  Budget.RowUpperBound.branch row10853_large_right_left_covers row10853_large_right_right_covers

def row10853_large : List Nat := row10853_large_left ++ row10853_large_right
theorem row10853_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10853 row10853_large :=
  Budget.RowUpperBound.branch row10853_large_left_covers row10853_large_right_covers

def row10853 : List Nat := row10853_large
theorem row10853_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10853 row10853 := row10853_large_covers

def row10859_small : List Nat := []
theorem row10859_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10859 row10859_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10859_block_005 : List Nat := []
theorem row10859_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10859 row10859_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10859_block_006 : List Nat := [11471, 11987, 12143, 12203]
def plan10859_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 29), .second, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 13), .second, .first, .first, .first, .first, .possible, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.secondFactor 17), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 23), .first, (.firstFactor 31), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, .first, .second, .possible, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10859_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10859 row10859_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10859_block_006
  · decide
  · rfl

def row10859_block_007 : List Nat := [13463, 13691, 14033]
def plan10859_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 37), .first, .first, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 23), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10859_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10859 row10859_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10859_block_007
  · decide
  · rfl

def row10859_block_008 : List Nat := [14783, 14957, 15161]
def plan10859_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), (.secondFactor 19), (.firstFactor 7), .second, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 37), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .possible, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10859_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10859 row10859_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10859_block_008
  · decide
  · rfl

def row10859_block_009 : List Nat := []
theorem row10859_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10859 row10859_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10859_block_010 : List Nat := []
theorem row10859_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10859 row10859_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10859_block_011 : List Nat := []
theorem row10859_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10859 row10859_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10859_block_012 : List Nat := []
theorem row10859_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10859 row10859_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10859_block_013 : List Nat := []
theorem row10859_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10859 row10859_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10859_large_left_left : List Nat := row10859_small ++ row10859_block_005
theorem row10859_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10859 row10859_large_left_left :=
  Budget.RowUpperBound.branch row10859_small_covers row10859_block_005_covers

def row10859_large_left_right_right : List Nat := row10859_block_007 ++ row10859_block_008
theorem row10859_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10859 row10859_large_left_right_right :=
  Budget.RowUpperBound.branch row10859_block_007_covers row10859_block_008_covers

def row10859_large_left_right : List Nat := row10859_block_006 ++ row10859_large_left_right_right
theorem row10859_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10859 row10859_large_left_right :=
  Budget.RowUpperBound.branch row10859_block_006_covers row10859_large_left_right_right_covers

def row10859_large_left : List Nat := row10859_large_left_left ++ row10859_large_left_right
theorem row10859_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10859 row10859_large_left :=
  Budget.RowUpperBound.branch row10859_large_left_left_covers row10859_large_left_right_covers

def row10859_large_right_left : List Nat := row10859_block_009 ++ row10859_block_010
theorem row10859_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10859 row10859_large_right_left :=
  Budget.RowUpperBound.branch row10859_block_009_covers row10859_block_010_covers

def row10859_large_right_right_right : List Nat := row10859_block_012 ++ row10859_block_013
theorem row10859_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10859 row10859_large_right_right_right :=
  Budget.RowUpperBound.branch row10859_block_012_covers row10859_block_013_covers

def row10859_large_right_right : List Nat := row10859_block_011 ++ row10859_large_right_right_right
theorem row10859_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10859 row10859_large_right_right :=
  Budget.RowUpperBound.branch row10859_block_011_covers row10859_large_right_right_right_covers

def row10859_large_right : List Nat := row10859_large_right_left ++ row10859_large_right_right
theorem row10859_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10859 row10859_large_right :=
  Budget.RowUpperBound.branch row10859_large_right_left_covers row10859_large_right_right_covers

def row10859_large : List Nat := row10859_large_left ++ row10859_large_right
theorem row10859_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10859 row10859_large :=
  Budget.RowUpperBound.branch row10859_large_left_covers row10859_large_right_covers

def row10859 : List Nat := row10859_large
theorem row10859_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10859 row10859 := row10859_large_covers

def row10861_small : List Nat := []
theorem row10861_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10861 row10861_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10861_block_005 : List Nat := []
theorem row10861_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10861 row10861_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10861_block_006 : List Nat := [10867, 11287, 11701]
def plan10861_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .possible, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .second, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 17), (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row10861_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10861 row10861_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10861_block_006
  · decide
  · rfl

def row10861_block_007 : List Nat := [12457, 12799, 13417]
def plan10861_block_007 : List Budget.Choice := [(.secondFactor 29), (.firstFactor 7), .first, .first, .possible, .first, .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 37), (.firstFactor 13), .first, .possible, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row10861_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10861 row10861_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10861_block_007
  · decide
  · rfl

def row10861_block_008 : List Nat := [14923]
def plan10861_block_008 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 37), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 37), .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10861_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10861 row10861_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10861_block_008
  · decide
  · rfl

def row10861_block_009 : List Nat := []
theorem row10861_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10861 row10861_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10861_block_010 : List Nat := []
theorem row10861_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10861 row10861_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10861_block_011 : List Nat := []
theorem row10861_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10861 row10861_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10861_block_012 : List Nat := []
theorem row10861_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10861 row10861_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10861_block_013 : List Nat := []
theorem row10861_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10861 row10861_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10861_large_left_left : List Nat := row10861_small ++ row10861_block_005
theorem row10861_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10861 row10861_large_left_left :=
  Budget.RowUpperBound.branch row10861_small_covers row10861_block_005_covers

def row10861_large_left_right_right : List Nat := row10861_block_007 ++ row10861_block_008
theorem row10861_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10861 row10861_large_left_right_right :=
  Budget.RowUpperBound.branch row10861_block_007_covers row10861_block_008_covers

def row10861_large_left_right : List Nat := row10861_block_006 ++ row10861_large_left_right_right
theorem row10861_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10861 row10861_large_left_right :=
  Budget.RowUpperBound.branch row10861_block_006_covers row10861_large_left_right_right_covers

def row10861_large_left : List Nat := row10861_large_left_left ++ row10861_large_left_right
theorem row10861_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10861 row10861_large_left :=
  Budget.RowUpperBound.branch row10861_large_left_left_covers row10861_large_left_right_covers

def row10861_large_right_left : List Nat := row10861_block_009 ++ row10861_block_010
theorem row10861_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10861 row10861_large_right_left :=
  Budget.RowUpperBound.branch row10861_block_009_covers row10861_block_010_covers

def row10861_large_right_right_right : List Nat := row10861_block_012 ++ row10861_block_013
theorem row10861_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10861 row10861_large_right_right_right :=
  Budget.RowUpperBound.branch row10861_block_012_covers row10861_block_013_covers

def row10861_large_right_right : List Nat := row10861_block_011 ++ row10861_large_right_right_right
theorem row10861_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10861 row10861_large_right_right :=
  Budget.RowUpperBound.branch row10861_block_011_covers row10861_large_right_right_right_covers

def row10861_large_right : List Nat := row10861_large_right_left ++ row10861_large_right_right
theorem row10861_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10861 row10861_large_right :=
  Budget.RowUpperBound.branch row10861_large_right_left_covers row10861_large_right_right_covers

def row10861_large : List Nat := row10861_large_left ++ row10861_large_right
theorem row10861_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10861 row10861_large :=
  Budget.RowUpperBound.branch row10861_large_left_covers row10861_large_right_covers

def row10861 : List Nat := row10861_large
theorem row10861_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10861 row10861 := row10861_large_covers

def row10867_small : List Nat := []
theorem row10867_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10867 row10867_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10867_block_005 : List Nat := []
theorem row10867_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10867 row10867_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10867_block_006 : List Nat := []
def plan10867_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 23), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 37), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 29), (.secondFactor 37), (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first]
set_option maxRecDepth 2048 in
theorem row10867_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10867 row10867_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10867_block_006
  · decide
  · rfl

def row10867_block_007 : List Nat := [12589]
def plan10867_block_007 : List Budget.Choice := [.first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .possible, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 31), .second, (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 37), (.firstFactor 11), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10867_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10867 row10867_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10867_block_007
  · decide
  · rfl

def row10867_block_008 : List Nat := []
def plan10867_block_008 : List Budget.Choice := [(.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .second, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10867_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10867 row10867_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10867_block_008
  · decide
  · rfl

def row10867_block_009 : List Nat := []
theorem row10867_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10867 row10867_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10867_block_010 : List Nat := []
theorem row10867_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10867 row10867_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10867_block_011 : List Nat := []
theorem row10867_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10867 row10867_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10867_block_012 : List Nat := []
theorem row10867_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10867 row10867_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10867_block_013 : List Nat := []
theorem row10867_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10867 row10867_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10867_large_left_left : List Nat := row10867_small ++ row10867_block_005
theorem row10867_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10867 row10867_large_left_left :=
  Budget.RowUpperBound.branch row10867_small_covers row10867_block_005_covers

def row10867_large_left_right_right : List Nat := row10867_block_007 ++ row10867_block_008
theorem row10867_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10867 row10867_large_left_right_right :=
  Budget.RowUpperBound.branch row10867_block_007_covers row10867_block_008_covers

def row10867_large_left_right : List Nat := row10867_block_006 ++ row10867_large_left_right_right
theorem row10867_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10867 row10867_large_left_right :=
  Budget.RowUpperBound.branch row10867_block_006_covers row10867_large_left_right_right_covers

def row10867_large_left : List Nat := row10867_large_left_left ++ row10867_large_left_right
theorem row10867_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10867 row10867_large_left :=
  Budget.RowUpperBound.branch row10867_large_left_left_covers row10867_large_left_right_covers

def row10867_large_right_left : List Nat := row10867_block_009 ++ row10867_block_010
theorem row10867_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10867 row10867_large_right_left :=
  Budget.RowUpperBound.branch row10867_block_009_covers row10867_block_010_covers

def row10867_large_right_right_right : List Nat := row10867_block_012 ++ row10867_block_013
theorem row10867_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10867 row10867_large_right_right_right :=
  Budget.RowUpperBound.branch row10867_block_012_covers row10867_block_013_covers

def row10867_large_right_right : List Nat := row10867_block_011 ++ row10867_large_right_right_right
theorem row10867_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10867 row10867_large_right_right :=
  Budget.RowUpperBound.branch row10867_block_011_covers row10867_large_right_right_right_covers

def row10867_large_right : List Nat := row10867_large_right_left ++ row10867_large_right_right
theorem row10867_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10867 row10867_large_right :=
  Budget.RowUpperBound.branch row10867_large_right_left_covers row10867_large_right_right_covers

def row10867_large : List Nat := row10867_large_left ++ row10867_large_right
theorem row10867_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10867 row10867_large :=
  Budget.RowUpperBound.branch row10867_large_left_covers row10867_large_right_covers

def row10867 : List Nat := row10867_large
theorem row10867_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10867 row10867 := row10867_large_covers

end ProjectEulerSolutions.PrimePairLarge
