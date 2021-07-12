install ()
{
    #VERSION=$(git describe)
    echo "install VERSION=$VERSION PREFIX=$PREFIX"
    
    if test -z "$SCONSUTILS_DIR"; then
        mv SConstruct SConstruct-disabledForIcs 2>/dev/null
    fi

    default_install
}

