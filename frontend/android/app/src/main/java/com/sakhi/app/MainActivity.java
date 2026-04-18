package com.sakhi.app;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(CactusPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
